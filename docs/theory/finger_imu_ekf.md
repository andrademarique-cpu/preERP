# EKF sobre un dedo de 2 falanges con IMUs — reporte del notebook

Documenta qué hace y qué demuestra
[`notebooks/finger_imu_practice.ipynb`](../../notebooks/finger_imu_practice.ipynb).

Trabajo exploratorio: valida sobre MuJoCo las decisiones de diseño que después
tiene que absorber el paquete `erp`. No es código de producción y no forma parte
del paquete.

---

## 1. Qué problema resuelve

Un dedo robótico de dos falanges con una IMU (acelerómetro + giróscopo) en cada
falange. Dos preguntas:

1. **¿Dónde está el dedo?** Estimar $[q_1, q_2, \dot q_1, \dot q_2]$ fusionando
   las dos IMUs con un EKF.
2. **¿Cuánto vale esa estimación?** Propagar la incertidumbre hasta la posición
   de la punta y verificar que la barra de error resultante es honesta.

La segunda pregunta es la que manda. Una posición sin covarianza confiable no es
una estimación, es una adivinanza — y el notebook muestra un caso concreto donde
la posición parece buena y la covarianza es ficción.

---

## 2. El pipeline de un vistazo

```
assets/finger_2link.xml  (template str.format, NO es XML cargable)
        │
        ├── build_xml(motor_lag=True)  ──> model_sim  (planta, 6 estados)
        └── build_xml(motor_lag=False) ──> model_ekf  (filtro, 4 estados)
                                                │
   trayectoria raised-cosine ──> plant_run() ──> X, Z, U  (verdad + IMU + control)
                                                │
   mjd_transitionFD ──> f, F, h, H  ────────────┤
                                                ▼
                                          EKF (Joseph)
                                                │
                          ┌─────────────────────┴──────────────────────┐
                          ▼                                            ▼
                   NEES / NIS                                 fk_tip + J P Jᵀ
              (¿es consistente?)                        (¿dónde está la punta y
                                                          con qué elipse?)
```

---

## 3. Bloque por bloque

### 3.1 Parámetros — el XML es solo topología

Regla que sostiene todo el resto: **si un número tiene significado físico, vive
en un dict de Python, no en el XML.** El XML es un template `str.format` con
slots; cargarlo directo con `mj.MjModel.from_xml_path` falla a propósito.

Esto permite barrer un parámetro (ganancia, largo de falange, $\tau$ del motor)
sin tocar el template, y compilar dos variantes desde una sola fuente de verdad.

| grupo | contenido |
|---|---|
| `geometry` | base 30 mm, falange proximal 35 mm, distal 30 mm; IMU en el punto medio de cada falange; damping estructural 0.002 N·m·s/rad |
| `sim_options` | `timestep` 2 ms, gravedad −9.81 m/s², integrador `implicitfast` |
| `joints` | recorrido mecánico: joint_1 ±90°, joint_2 ±120° |
| `actuators` | servos de posición: `kp`, `kv`, torque de stall, `armature` (inercia del rotor reflejada), damping de reductora, y `tau = 4 ms` |

`implicitfast` no es cosmético: con Euler explícito, ganancias de servo realistas
vuelven la simulación inestable.

### 3.2 Modelo dual — la decisión central

Un mismo template compila dos modelos:

| | `model_sim` (planta) | `model_ekf` (filtro) |
|---|---|---|
| dinámica del actuador | `dyntype="filterexact"`, $\tau = 4$ ms | ninguna |
| `na` | 2 | 0 |
| `nx = 2n_v + n_a` | **6** — $[q_1,q_2,v_1,v_2,a_1,a_2]$ | **4** — $[q_1,q_2,v_1,v_2]$ |
| rol | genera datos sintéticos de IMU | provee los jacobianos |

**Por qué dos modelos y no uno de 6×6 recortado a 4×4.** En la planta el control
$u$ entra *solo* por las activaciones $a$; el cuerpo rígido ve $a$, no $u$.
Recortar $B_{6\times2}$ a sus primeras cuatro filas no da el $B$ del cuerpo
rígido: da ese $B$ escalado por la ganancia de un paso
$1 - e^{-\Delta t/\tau}$. El notebook lo verifica numéricamente:

```
B_sim[:4] / B_ekf  ~  0.393469
1 - exp(-dt/tau)   =  0.393469     <- coinciden: el recorte es B mal escalada
```

Y además recortar $A$ tira las columnas de $a$, que llevaban el resto del efecto.
El par recortado es un modelo **mal**, no un modelo aproximado.

### 3.3 Datos sintéticos

Referencia por joint: coseno invertido (*raised cosine*),
$\theta(t) = \frac{A}{2}(1 - \cos\omega t)$, elegido para que **ángulo y
velocidad arranquen ambos en 0** — coherente con el estado que deja
`mj_resetData`, sin pico de torque al arrancar.

- joint_1: 45° a 0.5 Hz · joint_2: 60° a 0.3 Hz · 6 s de simulación
- `data_sim.ctrl` es el **ángulo objetivo en radianes**: el PD vive en el
  actuador de MuJoCo, no en un lazo hecho a mano en Python
- Los datos salen **siempre** de `model_sim`; `model_ekf` nunca genera datos

### 3.4 Linealización y validación de `A`

`mjd_transitionFD` devuelve cuatro jacobianos en un solo llamado:

$$A = \frac{\partial x_{k+1}}{\partial x},\quad
B = \frac{\partial x_{k+1}}{\partial u},\quad
C = \frac{\partial z}{\partial x},\quad
D = \frac{\partial z}{\partial u}$$

Que `A` tenga la forma correcta no prueba nada, así que hay un bloque de
validación entero. Tres cosas que ese test tiene que respetar:

1. **La verdad de referencia es `model_ekf`, no `model_sim`.** Medir contra la
   planta mezcla error de linealización con el lag de 4 ms — dos problemas con
   dos remedios distintos (relinealizar vs. inflar $Q$).
2. **`A` es un jacobiano: el modelo está en desviaciones.** El término afín no es
   decoración, porque $x_{op}$ **no** es un equilibrio: el servo está quieto pero
   la gravedad sigue tirando, así que $f(x_{op},u_{op}) \neq x_{op}$.
3. **"Error chico" no significa nada sin escala.** A 2 ms el estado casi no se
   mueve, así que *cualquier* modelo — incluso uno muy malo — anota un error
   absoluto chico. Se reporta relativo al movimiento real por paso.

Dos modos de propagación, porque falsan cosas distintas:

| modo | se le da | detecta |
|---|---|---|
| **un paso** | el $x_{k-1}$ real, siempre | error de modelo sin acumulación |
| **lazo abierto** | solo $x_0$, después su propia salida | las filas de **velocidad**, que el error en $q$ solo **no** puede falsar |

Resultados, con controles deliberadamente corrompidos:

| modelo | error en q por paso | divergencia en lazo abierto |
|---|---|---|
| `A`, `B` correctas | **0.275 %** | **2.6e-02 °** |
| MAL: `A = I` | 2679 % | 9.8e+03 ° |
| MAL: `A` escalada 1 % | 351 % | 5.6e+00 ° |
| MAL: `A[2,0] = 0` | **0.275 %** ← idéntico | 2.2e+01 ° (~850× peor) |
| MAL: `B = 0` | 2689 % | 7.5e+01 ° |

La fila `A[2,0]=0` es el argumento entero: es una fila de **velocidad**, y a un
paso no toca $q$. Validar solo $q$ deja medio `A` sin testear. Por eso están los
dos modos.

El test más fuerte es el de perturbación — comprueba la definición misma de
derivada, $f(x_{op}+dx) - f(x_{op})$ contra $A\,dx$:

| $\lVert dx \rVert$ | error relativo mediano |
|---|---|
| 1e-06 | 3.00e-09 |
| 1e-04 | 5.67e-07 |
| 1e-02 | 4.44e-05 |
| 1e-01 | 4.66e-04 |

El error cae linealmente con $\lVert dx \rVert$: eso es el término de segundo
orden, o sea `A` **es** exactamente la derivada.

Dato adicional: radio espectral de `A` = 0.8550 (< 1), así que el error de
linealización se amortigua en vez de acumularse.

### 3.5 Modelo de medición: `h` y `H` separados

Van aparte a propósito: el filtro las llama a ritmos distintos — `h` en cada
innovación (un `mj_forward`), `H` solo al relinealizar ($2n_x$ evaluaciones por
diferencias finitas). Fusionarlas obliga a pagar `H` cada vez que uno solo quería
`z_pred`.

Tres errores fáciles, los tres verificados en el notebook:

- **`h` toma `u`, no solo `x`.** Los acelerómetros leen `qacc`, y `qacc` depende
  del torque del servo. $D = \partial h/\partial u$ es no nulo en los
  acelerómetros y exactamente cero en giróscopos y sensores de posición — así que
  una innovación armada como $z - h(x)$ queda sesgada en los canales de
  acelerómetro.
- **`mj_forward`, no una etapa más barata.** Los sensores `mjSENS_ACC` recién son
  válidos después de la etapa de aceleración; parar en
  `mj_fwdPosition + mj_sensorPos` deja los acelerómetros mal mientras los otros
  cuatro sensores ya se ven bien.
- **`H` es la `C` de `mjd_transitionFD`, no las `DsDq` de `mjd_inverseFD`.** Estas
  últimas tratan $(q,v,a)$ como independientes e ignoran $\partial a/\partial q$.
  Para la fila `acc_i1_y` eso es **9.22 contra un valor real de 716.35**.

Gotcha relacionado: **`mj_step` no refresca `sensordata`**. Después de
`mj_step`, `data.sensordata` todavía tiene las lecturas del estado
*pre-integración*. Hay que llamar `mj_forward` después del paso antes de loguear
sensores, o loguear antes de integrar.

### 3.6 El EKF

Escrito contra un *argumento de modelo*, no contra `model_ekf`, así que la misma
clase corre sobre cualquiera de los dos modelos compilados. Eso **es** el
experimento.

- `f_dyn` / `F_dyn` — predicción (`mj_step` y el bloque `A` de `mjd_transitionFD`)
- `h_dyn` / `H_dyn` — medición (`mj_forward` y el bloque `C`)
- Update en **forma de Joseph**
- Sensores fusionados: solo IMU (`acc_i1`, `gyro1`, `acc_i2`, `gyro2`). Los
  sensores de posición quedan **fuera** a propósito — son la verdad de referencia.

Tres cosas que costaron depuración real:

- **Alineación temporal.** `X[k]`, `Z[k]` y `U[k]` tienen que referirse al mismo
  instante, así que el loop de planta loguea *antes* de `mj_step` y
  $X[k+1] = f(X[k], U[k])$ vale exacto. Loguear después reusando el `u` anterior
  desincroniza todo un paso — vale ~0.13° acá, que es el movimiento por paso
  entero, y se disfraza de error de modelo.
- **Ruido de proceso conocido.** Con el modelo matched la predicción es *exacta*,
  así que $Q \to 0$, `P` colapsa y el NEES explota por puro redondeo. Un test de
  consistencia necesita una perturbación de covarianza conocida: la planta recibe
  una perturbación de torque explícita y $Q$ se construye para calzar con ella
  (modelo DWNA, $G = [\Delta t^2/2\;I;\; \Delta t\, I;\; 0]$, que da la
  correlación q-v correcta en vez de dos bloques diagonales sueltos).
- **Mantener `P` definida positiva.** `F` tiene entradas de orden 700 y `P` abarca
  diez órdenes de magnitud entre el bloque de posición y el de velocidad;
  $F P F^T$ pierde simetría y definición positiva en punto flotante. **Sin el piso
  de autovalores el NEES sale negativo** y el diagnóstico deja de significar algo.

---

## 4. Resultado principal: matched vs. mismatched

2000 pasos, la misma trayectoria y los mismos datos para los dos filtros. Lo
único que cambia es qué modelo se le da al EKF.

| | matched (`model_sim`, 6 est.) | mismatched (`model_ekf`, 4 est.) |
|---|---|---|
| sesgo de medición \|acc_i2_y\| | **0.0000** m/s² | **4.0101** m/s² (σ sensor = 0.05 → ~80σ) |
| NEES mediana | **5.96** (objetivo 6) ✅ | **344 281** (objetivo 4) ❌ |
| NIS media | **12.03** (objetivo 12) ✅ | 23.29 ❌ |
| RMS q | 0.0030°, 0.0027° | 0.153°, 0.107° |
| σ reportado por `P` | 0.0022°, 0.0030° | 0.0004°, 0.0013° |
| cobertura ±2σ en q1 | 78.9 % | 0.1 % |

**El filtro matched es consistente.** NIS sobre su objetivo y NEES mediana sobre
$n_x$. Eso es el filtro funcionando: `f`, `F`, `h`, `H`, el update de Joseph y la
contabilidad de $Q$/$R$ son todos mutuamente correctos.

**El mismatched no lo es, y inflar $Q$ no lo salva.** Este es el hallazgo del
notebook, y contradice lo que decía la propia celda de diseño del modelo dual:

> El sesgo es *determinista y correlacionado en el tiempo*: el acelerómetro lee
> `qacc`, `qacc` sale del torque del servo, y el torque sale de la activación que
> `model_ekf` no tiene. $Q$ describe ruido de proceso blanco y de media cero, así
> que puede ensanchar `P` pero **nunca** quitar un sesgo en `h`.

Inflar $R$ en los canales de acelerómetro aplana el NIS pero deja el NEES
intacto, porque el error no es blanco: `P` sigue encogiéndose como si la
información fuera independiente cuando en realidad es el mismo sesgo llegando
500 veces por segundo.

Opciones que **sí** funcionan:

1. **Aumentar el estado** con la activación — lo que hace el filtro matched acá.
2. **Sacar los acelerómetros** y fusionar solo giróscopos: leen `qvel`, que es un
   estado, así que su $D = \partial h/\partial u$ es exactamente cero y no
   arrastran sesgo por lag.
3. Modelar el sesgo explícitamente como un estado de random walk.

> **Nota de lectura del NEES.** Se reporta la **mediana**, no la media: la
> distribución tiene cola pesada (el percentil 99 está órdenes por encima de la
> mediana) por momentos breves donde `P` es chica y la linealización es pobre. La
> mediana es el estadístico robusto para una corrida única; un ANEES en serio
> necesita Monte Carlo sobre semillas independientes.

---

## 5. De los ángulos a la punta

El EKF estima ángulos. Lo que el resto del sistema quiere es **dónde está la
punta**, con su incertidumbre.

El dedo se mueve en el plano y-z (ambas bisagras tienen `axis="1 0 0"`):

$$y = -\left(l_1\sin q_1 + l_2\sin(q_1{+}q_2)\right), \qquad
  z = l_0 + l_1\cos q_1 + l_2\cos(q_1{+}q_2)$$

$$\Sigma = J\,P_{qq}\,J^{T}, \qquad J = \frac{\partial(y,z)}{\partial(q_1,q_2)}$$

Verificación de la cinemática (contra el sensor `framepos` de MuJoCo, que **no**
está en las filas que el EKF actualiza, así que no contamina el filtro):

```
fk_tip vs framepos de MuJoCo:  max |dif| = 2.8e-17 m
J analitico vs mj_jacSite:     max |dif| = 2.8e-17 m/rad
J analitico vs dif. finitas:   max |dif| = 2.9e-12 m/rad
```

Escala útil para la reunión: **1° de error en $q_1$ mueve la punta 1.125 mm**;
1° en $q_2$, 0.524 mm. La punta recorre 54.7 mm en y y 39.0 mm en z.

### Resultados en la punta

| | matched | mismatched |
|---|---|---|
| error de la punta RMS | **2.44 µm** (max 11.1) | **139.4 µm** (max 322) |
| σ propagada RMS | 2.58 µm | 0.48 µm |
| NEES de la punta (mediana) | 1.71 (χ²(2): mediana 1.386) | 333 699 |
| cobertura elipse 95 % | 85.0 % | **0.0 %** |

**El filtro matched ubica la punta con ~2.6 µm sobre un recorrido de 55 mm** —
0.005 % del travel.

**Para el mismatched la elipse es ficción.** Error de 139 µm contra un σ
propagado de 0.5 µm: exceso de NEES de 5 órdenes, cobertura 0 %. Propagar una
covarianza desde un filtro sesgado da una respuesta **confiadamente equivocada**,
que es peor que no tener covarianza: el número parece usable.

### Cuatro cosas que este bloque verifica en vez de asumir

1. **Solo entra el bloque `q` de `P`.** La punta es función de $q$ sola; $v$ no
   entra. Propagar la `P` completa sería incorrecto, no meramente derrochador.

2. **`P_qq` entera, no su diagonal.** Los mismos acelerómetros informan $q_1$ y
   $q_2$, así que sus errores están correlacionados ($\rho \approx -0.30$).
   Tirando el término cruzado:

   | métrica | efecto |
   |---|---|
   | área de la elipse | 14.29 → 14.95 µm² (solo **1.05×**) |
   | σ en la **peor dirección** | **1.19×** (max 1.2×) |

   El área casi no cambia, factor $1/\sqrt{1-\rho^2}$ — o sea que **mirar el área
   oculta el error**. Lo que se rompe es la *orientación* de la elipse. Es
   exactamente el error que comete quien guarda `sqrt(diag(P))` y lo llama la
   incertidumbre.

3. **La propagación es un cambio de coordenadas, no una fuente nueva de error.**
   $J$ es invertible, así que el NEES en espacio de la punta tiene que ser igual
   al NEES en espacio de ángulos, a precisión de máquina. Medido: **1.0e-09**.
   Ese es el test más filoso disponible acá, y es el que permite localizar la
   culpa: como los dos NEES coinciden, la cobertura de 85 % (en vez de 95 %) es
   un diagnóstico **del filtro**, no de la cinemática. El bloque marginal `q` está
   ~1.22× sobreconfiado aunque el estado *entero* sea consistente.

   > Consistencia del estado completo **no** implica consistencia marginal — y es
   > la marginal la que le importa a quien consume la posición de la punta.

4. **El primer orden es una afirmación sobre curvatura, no una ley.** Se mide
   contra sigma-points, inflando `P` a propósito hasta romperlo:

   | `P` × | σ_q1 | error vs unscented |
   |---|---|---|
   | 1e0 (real) | 0.0026° | 1.90e-09 |
   | 1e2 | 0.026° | 1.91e-07 |
   | 1e4 | 0.26° | 1.91e-05 |
   | 1e6 | 2.60° | 1.90e-03 |
   | 1e7 | 8.22° | 1.89e-02 |

   El error escala **lineal** con $\lVert P \rVert$: la firma de un término de
   segundo orden bien calculado. Recién llega a ~2 % con σ_q ≈ 8°, unas 3000×
   la incertidumbre real del filtro. **Conclusión: una UKF del lado de la salida
   no compraría nada en este punto de operación** — y eso es un resultado medido,
   no una suposición.

   (La columna de Monte Carlo se planta en $\sqrt{2/n_{mc}}$, su propio ruido de
   muestreo. El test fino es el unscented.)

### La elipse es casi degenerada

| | valor |
|---|---|
| semieje mayor (mediana) | 6.75 µm |
| semieje menor (mediana) | 0.78 µm |
| relación de aspecto | 7.9:1 mediana, **7663:1** máx |

Eso es geometría, no artefacto numérico: cerca de $q_2 = 0$ (dedo estirado) las
dos columnas de $J$ se vuelven paralelas — ambos joints empujan la punta casi en
la misma dirección — y la incertidumbre colapsa sobre una recta. Un escalar
"±3 µm" tira esa información a la basura; la elipse dice **además hacia dónde**.
Es también la razón de usar `solve` en vez de `inv`: `cond(S)` llega a 1e11.

---

## 6. Qué se lleva el paquete `erp`

Lo que este notebook deja probado y debería condicionar el diseño de
`software/src/erp/`:

- **El sesgo de modelo no se arregla con $Q$.** Si el estimador no modela la
  dinámica del actuador, hay que aumentar el estado o descartar los sensores
  contaminados. Inflar el ruido produce un filtro que *parece* calibrado por NIS
  y sigue siendo inconsistente por NEES.
- **NEES/NIS es el criterio, no la inspección visual.** Las dos trayectorias de
  este notebook se ven razonables graficadas; solo el NEES separa 5.96 de
  344 281. Esto ya está en `CLAUDE.md` como regla y acá está el caso concreto.
- **La covarianza se propaga entera, no por su diagonal.** El área casi no acusa
  el error; la orientación sí.
- **`FusionEngine` tiene que preservar la alineación temporal.** El desfase de un
  paso vale acá el movimiento por paso entero y se disfraza de error de modelo.
- **El piso de autovalores sobre `P` no es cosmético.** Sin él el NEES sale
  negativo y el diagnóstico se rompe en silencio.
- La UKF no se justifica *en este punto de operación* por curvatura de la salida
  — si entra al stack, que sea por otra razón y medida.

---

## 7. Cómo correrlo

```bash
# desde la raiz del repo
jupyter lab notebooks/finger_imu_practice.ipynb
```

Requiere `mujoco`, `numpy`, `scipy` y `matplotlib`. El notebook resuelve solo la
ruta al template (`resolve_model_path`), así que corre tanto desde la raíz del
repo como desde `notebooks/`.

Estructura: 31 celdas, ~6 s de simulación a 2 ms y 2000 pasos de EKF por cada uno
de los dos modelos. Los bloques de control (sigma-points, Monte Carlo, barrido de
`P`) son la parte más lenta.

### Parámetros de ruido

| símbolo | valor | qué es |
|---|---|---|
| `SIG_ACC` | 0.05 m/s² | ruido de acelerómetro |
| `SIG_GYR` | 0.005 rad/s | ruido de giróscopo |
| `SIG_ALPHA` | 12 rad/s² | perturbación de torque (1σ), lo que describe $Q$ |
| `SIG_ACT` | 2e-4 rad | jitter de la activación del servo |
