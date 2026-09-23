# `scripts/demo_teleop.py` — la tuberia completa

Recorrido detallado de la demo interactiva: que corre, en que orden, con que
relojes, y por que cada numero es el que es.

**Castellano**, siguiendo al archivo que documenta (`scripts/demo_teleop.py`) y
al lado de fisica-y-filtrado del corte de idiomas de CLAUDE.md. Los modulos de
plomeria que aparecen aca (`erp.fusion`, `erp.sensors`) son ingles en el codigo;
sus nombres se citan tal cual.

---

## 1. La tesis, en una linea

**El EKF es ciego a la consigna por construccion.** `EKF` no tiene donde meter
una `u` — `self.u_blind` es un vector de ceros alocado una vez, y ni `predict`
ni `update` toman un parametro de control. Un test lo afirma por firma.

Entonces cuando movés una junta con el teclado, el filtro **no ve la tecla**.
Estima las activaciones de los tres servos a partir de dos IMUs ruidosas y nada
mas. El fantasma se atrasa y despues alcanza a la verdad, y eso que se ve es el
filtro *infiriendo* un comando que nunca recibio. Un barrido sinusoidal
graficado despues no puede mostrar eso.

## 2. Las tres senales

Son de la **misma magnitud fisica** y no son intercambiables. Confundirlas
produce una demo convincente que no significa nada.

| Senal | De donde sale | Nota |
|---|---|---|
| **verdad** | el modelo de planta SIN editar (`load_model`), integrado bajo tu consigna | `na = 0`. `test_plant.py` afirma que la planta *falla* el contrato `na == 3` del filtro |
| **medido** | `LiveSimSensor` muestreando esa planta a ~20 Hz, con ruido N(0, R) y latencia | indices `rows` dentro de `data.sensordata` |
| **estimado** | `h(x̂)` del EKF ciego, que corre sobre `blind_variant` | `na = 3`. Nunca ve `ctrl` |

## 3. Mapa de la tuberia

```
  teclado (GLFW key_callback)
      |  Teleop.on_key  -> held{idx: signo}
      v
  Teleop.jog()                        1 vez por tick de render (60 Hz)
      |  ctrl += signo * JOG_RAD_S / RENDER_HZ,  recortado a actuator_ctrlrange
      v
  Teleop.step(n_steps = 8)            8 pasos de fisica por tick (dt = 2 ms)
      |
      +--> por CADA paso de fisica:
      |      data.ctrl[:3] = ctrl
      |      mj_step(model, data)          <- la VERDAD avanza
      |      t += dt                       <- la unica base de tiempo
      |      reading = data.sensordata     (+ swap si --degrade swap)
      |      sensor.poll(t, reading)       <- muestrea si toca; si no, None
      |
      +--> UNA vez por tick, despues del lazo de fisica:
             for m in sensor.drain():      <- libera lo que ya cumplio latencia
                 runner.ingest(m)          <- avanza HASTA m.timestamp y corrige
             runner.advance_to_safe(t)     <- alcanza el presente, con margen
      v
  fantasma + graficos
      ghost_data()  -> x_hat en data_b, mj_forward
      draw_ghost(viewer.user_scn, model_b, data_b)
      viewer.sync()
      plots.push_plant / push_estimate,  redraw cada 2 ticks (30 Hz)
```

---

## 4. Arranque: `Teleop.__init__`

### 4.1 Dos modelos, no uno

```python
self.model,   self.data   = load_model(xml_path)      # planta: na = 0, nx = 8
self.model_b, self.data_b = blind_variant(xml_path)   # ciego:  na = 3, nx = 11
```

`blind_variant` compila el XML por `MjSpec` y le pone `dyntype =
mjDYN_INTEGRATOR` a los tres actuadores. Con los `<position>` originales y
`ctrl = 0`, el servo tira cada junta hacia cero y el filtro "sabria" que el
brazo vuelve a home. Con `integrator`, cada servo gana un estado de
**activacion** — el setpoint efectivo — que con `ctrl = 0` se queda quieto y `Q`
lo vuelve random walk. **Esa activacion es lo que el filtro estima en lugar del
comando que nunca recibe.**

Se compila por `MjSpec` y no editando texto: el XML tiene un `<include>` y
mallas con rutas relativas que `from_xml_string` no resuelve.

### 4.2 El estado del filtro

`nx = state_dim(model_b) = 2 * nv + na = 2*4 + 3 = 11`, ordenado

```
x = [ qpos[:nv] | qvel[:nv] | act[:na] ]
        4            4           3
```

Ojo con `nv` y no `nq`: las posiciones viven en el espacio **tangente**, que es
lo que hace valida la jacobiana por diferencias finitas de `mjd_transitionFD`.
En este brazo `nq == nv == 4`, asi que no se nota, pero la regla es esa.

### 4.3 Filas y `R`

```python
self.imu_rows       = rows_of(self.model_b, *cfg.imu.layout)   # 12 indices
self.imu_rows_plant = rows_of(self.model,   *cfg.imu.layout)
R_full = make_R(self.model_b, cfg.imu.sig_acc, cfg.imu.sig_gyro)
self.R = R_full[np.ix_(self.imu_rows, self.imu_rows)]
```

El cableado sale de `config/estimation.yaml` por `load_config` / `build_decoder`
— **la definicion unica desde P7**, IMU_0 hacia link1 e IMU_1 hacia link2. Una
quinta copia en la demo desharia eso. `build_decoder(cfg.imu)` se llama al
arrancar solo para **validar** el config antes de abrir una ventana.

Las 12 filas son 4 sensores por 3 ejes: `link1_acc`, `link2_acc`, `link1_gyro`,
`link2_gyro`. El `efector_pos` (filas 12-14) existe en el modelo pero **no** se
mide: es verdad de simulacion, no una IMU.

### 4.4 Estado inicial: el equilibrio servoado, no ceros

```python
self.q0 = self.ctrl.copy()                                  # [0, 0, 0] del keyframe
x_warm = np.r_[warmup_to_rest(self.model_b, self.data_b, self.q0), self.q0]
```

`warmup_to_rest` resetea al keyframe `home`, escribe la consigna y deja correr
100 pasos para que el brazo se asiente bajo su propio peso. Devuelve
`[qpos, qvel]` (8), y se le concatena `q0` como activacion inicial (3) → 11.

**Por que importa:** con `q = 0` y activacion 0 el servo no hace fuerza, la
gravedad acelera el brazo y `h(x0)` da `link2_acc_x` = **4.144** m/s² contra
**9.810** del equilibrio y ~9.65 en la IMU real. Sobre un canal cuya sigma es
0.05, esos 5.7 m/s² son un sesgo de **114 sigma** — ningun ajuste de `Q` o `R`
lo repara.

Cuidado con el contrafactico: poner el estado **entero** en cero da 345 m/s²,
porque eso ademas rompe la igualdad del tendon. Es una falla distinta y mucho
mas ruidosa; citarla exagera esta.

Ese sesgo de 114 sigma **no sobrevive la corrida**: el tendon devuelve el estado
en ~2 pasos, antes de la primera medicion a 5 ms. Por eso `--degrade zeros` da
NIS 5.70, bit-identico a sano.

### 4.5 `Q`, `P0` y la dinamica

```python
Q  = make_Q(self.model_b, SIG_ALPHA, SIG_ACT_BLIND, SIG_CM)
P0 = np.diag([np.deg2rad(1.0) ** 2] * self.nx)
self.ekf = EKF(x0, P0, Q, R_declared, MujocoDynamics(self.model_b, self.data_b))
```

| Constante | Valor | Que es |
|---|---|---|
| `SIG_ALPHA` | 12.0 rad/s² | 1 sigma de aceleracion por paso (DWNA) |
| `SIG_ACT_BLIND` | 5e-3 rad | random walk de la activacion, por paso |
| `SIG_CM` | 1e-4 rad | por paso a lo largo del modo comun |

Son **identicas** a las de `make_golden_run.py` y `demo_three_way.py` a
proposito: tres juegos de constantes serian tres experimentos distintos y las
corridas dejarian de ser comparables.

**`make_Q` arma una DWNA a un `dt` fijo de 2 ms y NO es invariante al
schedule**: solo vale al timestep del modelo. Los `psd_*` del config son PSD de
tiempo continuo (rad²/s³, rad²/s) del diseno viejo; convertir entre las dos
cosas es una re-derivacion, no un cambio de unidades.

---

## 5. El scheduler

### 5.1 Quien es dueno del lazo principal

`mujoco.viewer.launch_passive` **no bloquea**: devuelve un handle y uno llama
`.sync()` cuando quiere. Eso es lo que permite que el `QTimer` de Qt maneje todo
y no haya dos event loops peleando por el hilo principal.

```python
timer = QtCore.QTimer()
timer.timeout.connect(tick)
timer.start(int(1000.0 / RENDER_HZ))      # 16 ms
```

| Cantidad | Valor |
|---|---|
| Paso de fisica (`model.opt.timestep`) | 2 ms (500 Hz) |
| Tick de render (`RENDER_HZ`) | 60 Hz |
| Pasos de fisica por tick (`n_steps`) | `round((1/60) / 0.002)` = **8** |
| Redibujo de strips (`PLOT_HZ`) | 30 Hz — cada 2 ticks |
| IMU (`IMU_RATE_HZ`) | 20 Hz → ~25 pasos de fisica entre mediciones |
| Latencia de transporte (`IMU_LATENCY_S`) | 5 ms |

**Los ~25 predicts entre updates son el limite de precision actual**, no el
filtro.

### 5.2 La base de tiempo es `self.t`, y es la unica

`self.t` avanza `dt` por paso de fisica. Con ella sella el sensor, con ella
avanza el filtro y con ella se libera la latencia. **No se usa
`perf_counter()` en ninguna parte del lazo.**

Usar el reloj de pared para una parte y el de simulacion para otra pone cada
muestra en el pasado o en el futuro por la diferencia entre las dos bases, que
es constante y silenciosa.

`SimClock` es la envoltura que le da ese reloj al runner:

```python
class SimClock:
    def now(self) -> float:      return self._owner.t
    def sleep_until(self, t):    pass      # el que avanza el tiempo es la fisica
```

### 5.3 El orden dentro de un tick, que no es negociable

```python
def tick() -> None:
    if not viewer.is_running(): app.quit(); return
    _restore_view_options(viewer, view_opts)   # deshace los atajos del visor
    teleop.jog()                               # mueve la consigna
    y_plant, y_est = teleop.step(n_steps)      # fisica + sensor + filtro
    ...fantasma...; viewer.sync(); ...graficos...
```

Y dentro de `step`, lo importante: **drenar e ingerir UNA vez por tick, no una
vez por paso de fisica.**

Por paso, `advance_to_safe` correria 8 veces por tick y cada vez empujaria el
filtro hasta `t - horizonte`, que es justo el instante de la muestra que el
sensor esta por liberar: **el filtro le gana la carrera a su propia medicion.**

El orden entre las dos llamadas tambien importa: **primero ingerir** (cada
`_apply` avanza hasta SU sello) y **despues** alcanzar el presente.

---

## 6. El sensor virtual: `LiveSimSensor`

`SimSensor` no sirve para esto y la diferencia es estructural, no una opcion que
falta: `SimSensor` es un `ReplaySensor` sobre un array `(t, sensordata)`
**precomputado** cuyo ruido se sortea una sola vez al construirlo. La
teleoperacion no tiene log — la verdad se produce a medida que apretás teclas.

### 6.1 Muestreo sobre grilla absoluta

```python
due = self._t0 + self._k / self.rate_hz
if t + 1e-12 < due: return None
```

`t0 + k / rate_hz`, **no** `t + 1/rate_hz` acumulado. El segundo arrastra el
resto del paso de fisica y deriva — la misma falla que `RateLoop` existe para
evitar del lado de los comandos.

### 6.2 Ruido y entrega

```python
z = clean + L @ rng.standard_normal(k)        # L = sqrt_psd(R)
m = Measurement(z, t, rows, R, name)
```

`rows` y `R` son **un solo array compartido y de solo lectura** por sensor
(`shared_rows_R`), no una copia por muestra: quien mutara `m.R` in place
reescribiria en silencio el ruido de toda muestra pasada y futura. Un test
afirma que la escritura levanta excepcion.

La entrega respeta la latencia:

```python
def _release_end(self):
    if self.clock is None: return len(self._queue)
    return bisect_right(self._ts, self.clock() - self.latency_s, lo=self._cursor)
```

La demo pasa `clock=lambda: self.t`, o sea que una muestra tomada en `t` se
retiene hasta que el tiempo de **simulacion** llega a `t + 5 ms`.

**`Measurement.timestamp` es el instante de MUESTRA, no el de llegada.** Esa es
toda la razon por la que existe la seccion 9.

---

## 7. El EKF

`erp.estimators.ekf.EKF`, sobre una `DiscreteDynamics` (protocolo numpy puro).
Desde P4 **no importa mujoco**: sostiene la dinamica, no un `MjModel`.
`MujocoDynamics` es la implementacion con fisica; `LinearDynamics` es la que
permite compararlo contra un KF de solucion cerrada, que es la unica evidencia
real de que la aritmetica esta bien.

### 7.1 `predict()` — un paso de `dyn.dt`

```python
x_next, F = self.dyn.step(self.x, self.u_blind)
self.x = x_next
self.P = make_spd(F @ self.P @ F.T + self.Q)
```

`F` sale del **mismo** punto de linealizacion que `x_next` — por eso vienen
juntas de `step` y no de dos llamadas. Pedirlas por separado deja abierta la
posibilidad de evaluarlas en `x` distintos, y ese bug no se ve.

### 7.2 `update(z, rows, R)` — Joseph

```python
z_pred, H_full = self.dyn.observe(self.x, self.u_blind)
H = H_full[rows]
y = z[rows] - z_pred[rows]
S = make_spd(H @ P @ H.T + R_k)
K = np.linalg.solve(S, H @ P).T             # = P H^T S^-1
self.x = self.x + K @ y
I_KH = I - K @ H
self.P = make_spd(I_KH @ P @ I_KH.T + K @ R_k @ K.T)
return y, float(y @ np.linalg.solve(S, y))  # innovacion, NIS
```

Cuatro decisiones que cargan peso:

- **`make_spd` despues de cada operacion sobre `P`.** Simetriza y pisa los
  autovalores. Sin eso el NEES sale **negativo** y el diagnostico de
  consistencia falla en silencio.
- **Forma de Joseph** y no `(I-KH)P`. Medido sobre dos estados observados por su
  suma, con posterior de 8 ordenes de magnitud y `sigma_R = 1e-9`: la forma
  corta da autovalor minimo **−3.6e-17** (eso no es una covarianza) y Joseph
  **+5.0e-19**. Preciso: es UN update con una `R` extrema, no un filtro
  divergiendo a la vista.
- **`np.linalg.solve`, nunca una inversa explicita.** `S` se pone mal
  condicionada cuando el brazo se estira.
- **Un update de 12 canales, no cuatro de 3.** Medido: ~360 µs contra ~1.4 ms.
  Por eso una linea del sensor es exactamente una `Measurement`.

### 7.3 `observe` corre un segundo `mj_forward`, y no es redundante

`mjd_transitionFD` restaura `qpos`/`qvel`/`act` pero deja `sensordata` en su
ultima perturbacion de diferencias finitas. Leerlo directo da una `z`
equivocada por **4.3e-4** — como el 1% de un `sig_acc`, lo bastante chico para
parecer un problema de tolerancia. No lo "simplifiques".

### 7.4 `R` por medicion

`update` acepta una `R` (k, k) de la medicion y si no se la dan usa
`self.R[ix_(rows, rows)]`. Hoy los dos caminos dan el mismo numero — la `R` del
sensor se construye como ese mismo bloque — y eso es a proposito: mueve el
cableado sin mover la corrida congelada. Empieza a diferir cuando `calibrate()`
corra sobre el brazo de verdad.

### 7.5 Costo

| Operacion | Tiempo |
|---|---|
| predict + update apareados | ~131.5 µs |
| sin aparear (4 llamadas sueltas) | ~143.5 µs |
| costo por segundo de reloj | 500 x 131.5 µs ≈ **66 ms, ~7% de duty** |

Aparear las llamadas a MuJoCo ahorra **~8%, no 50%**: las cargas de `MjData` se
parten al medio, pero `mjd_transitionFD` corre `nx+1` evaluaciones internas y
ese es el costo real.

---

## 8. `FilterRunner`: de sellos de tiempo a pasos de filtro

`erp.fusion.runner` es el **unico** modulo que hace esa traduccion. Vive al lado
de `core/` y `io/`, no al lado de `estimators/`: decide **cuando** avanza el
filtro, nunca que calcula.

### 8.1 La aritmetica de paso

```python
def _step_of(self, t): return round((t - self.t0) / self._dt)
```

**Al mas cercano, no al piso.** A 2 ms de paso y 50 ms entre muestras, poner la
medicion en el paso mas cercano cuesta como mucho 1 ms de desalineacion contra
un hueco de 50 ms; el piso costaria hasta 2 ms y estaria sesgado tarde.

`t_filter = t0 + k * dt` — el filtro **existe solo en bordes de paso**, no en el
sello de la ultima medicion.

### 8.2 `ingest(m)`

```
horizonte <= 0  ->  _apply(m) directo
horizonte  > 0  ->  encolar en _pending
                    watermark = max(watermark, m.timestamp)
                    if clock: watermark = max(watermark, clock.now())
                    _release(watermark - horizonte)
```

`_release` ordena lo pendiente por sello y aplica todo lo que este en o antes
del limite.

### 8.3 `_apply(m)` y `discarded`

```python
if self._step_of(m.timestamp) < self._k:
    self.discarded += 1
    return None
self.advance_to(m.timestamp)
self._z_full[m.rows] = m.z
y, nis = self.est.update(self._z_full, m.rows, m.R)
```

Una medicion que redondea a un paso que el filtro **ya paso** se descarta y se
cuenta. No se levanta excepcion: un sensor vivo que produce una muestra tarde es
un evento normal, y un runner que tirara excepcion se llevaria puesto el lazo de
comandos. Tampoco se aplica: un update hacia atras no es un update de Kalman —
la covarianza que iba a corregir ya se propago mas alla.

**El criterio es el PASO, no el sello.** Una muestra sellada 0.9 ms antes del
tiempo actual del filtro sigue redondeando al paso en que el filtro esta, y se
aplica con cero predicts. El jitter sub-paso no es desorden, y contarlo como tal
reportaria drops sobre un stream que es meramente irregular.

> `discarded` es el numero que hay que mirar primero. Un `nis` que se ve sano
> porque la mayoria de las muestras nunca llego al filtro es exactamente lo que
> ese contador existe para hacer visible.

### 8.4 `_z_full`: un vector, reusado

El runner tiene un `_z_full` de `nz = 15` alocado una vez. Cada medicion escribe
sus 12 filas adentro y `EKF.update` lee solo las filas que le dan. Nunca se
realoca.

---

## 9. `buffer_horizon` — la parte que costo dos corridas

### 9.1 Que es

**Una pregunta, no dos: cuanto atras de tiempo real corre el estimado.**
Gobierna las dos cosas a la vez: cuando se libera una medicion retenida, y hasta
donde se anima a avanzar el filtro. Compra orden con latencia, y no hay una
tercera opcion.

`0.0` es correcto para un log reproducido y para una sola fuente sin latencia.
Se gana el sueldo cuando aparece un segundo stream con otra latencia — la
lectura de encoders del brazo, por ejemplo.

### 9.2 `advance_to_safe`, no `advance_to(now)`

```python
limit = t_now - self.buffer_horizon
if self._pending:
    limit = min(limit, min(m.timestamp for m in self._pending))
return self.advance_to(limit)
```

Avanzar hasta *ahora* parece correcto y es una trampa: el sensor sella el
instante de **muestra** y la entrega despues, asi que un filtro ya avanzado
hasta ahora recibe cada muestra sellada en su pasado y las tira todas.

La clausula extra — no pasar de lo que el buffer **todavia retiene** — evita que
un horizonte sin `clock` descarte todo lo que encola. El costo es que el
estimado se atrasa un periodo de muestreo extra en esa configuracion, que es el
precio de no inyectar un reloj, pagado visiblemente y no como un drop silencioso.

### 9.3 Las dos mitades del descubrimiento, medidas

| Configuracion | Resultado |
|---|---|
| `buffer_horizon = 0.0` | `advance_to_safe(t)` **es** `advance_to(t)`. El filtro corre adelante del sensor. **0 de 58 updates aplicados**, sin un solo error, con el visor andando y los graficos dibujando |
| `buffer_horizon = 0.005` (la latencia exacta) | 5 ms son **2.5** pasos de 2 ms. La marca de agua `t - horizonte` cae siempre en un empate de redondeo, y `round` de Python desempata **al par**. **41 de 58 descartadas, NIS mediana 6242** contra objetivo 12, el estimado sin seguir nada |
| `ceil(0.005 / 0.002) * 0.002 = 0.006` | La marca de agua cae en un borde de paso. **0 descartadas, NIS mediana 5.70, error terminal 0.003 rad** |

Por eso la demo escribe:

```python
horizon = np.ceil(IMU_LATENCY_S / self.dt) * self.dt      # 6 ms, no 5
```

**El horizonte tiene que ser un numero entero de pasos de fisica.** Las dos
fallas de arriba corrieron sin levantar una sola excepcion y con la ventana
dibujando, que es exactamente lo que las hace sobrevivir a una demo.

### 9.4 El `clock` acompana por obligacion

Con horizonte > 0 y **sin** reloj, la marca de agua es el sello mas nuevo
*ingerido*, o sea que cada muestra espera a su sucesora: **50 ms de atraso en
vez de 5**. Por eso el runner recibe `clock=SimClock(self)`.

```python
self.runner = FilterRunner(
    self.ekf, t0=0.0, buffer_horizon=float(horizon),
    clock=SimClock(self), record=False,
)
```

`record=False` apaga la traza de estado por paso, que es lo unico que crece con
la corrida — covarianzas `(n, 11, 11)` a 500 pasos por segundo. El NIS y sus
sellos **siempre** se guardan: dos floats por medicion, y el diagnostico que
decide si el filtro es consistente. Una bandera que apagara *eso* seria una
trampa.

### 9.5 Los tres ingredientes, juntos

Para que una fuente viva funcione hacen falta las tres, y cada una parecio exito
por separado:

1. `buffer_horizon > 0` — si no, el filtro se adelanta y descarta todo.
2. Redondeado **arriba** a un numero entero de pasos — si no, el empate de
   redondeo se come la mitad.
3. Drenar/ingerir **una vez por tick de render**, no una vez por paso de fisica.

---

## 10. El fantasma

```python
def ghost_data(self):
    self.data_b.qpos[:nv] = self.ekf.x[:nv]
    self.data_b.qvel[:nv] = self.ekf.x[nv:2*nv]
    if self.model_b.na: self.data_b.act[:] = self.ekf.x[2*nv:]
    mj.mj_forward(self.model_b, self.data_b)
    return self.data_b
```

`mj_forward` es obligatorio: `draw_ghost` lee `geom_xpos`/`geom_xmat`, que son
**salidas** del forward.

`draw_ghost` dibuja **25 de 27** geoms: `ghost_geom_ids` saltea los que estan
soldados al mundo (`body_weldid == 0`), o sea el piso de `scene.xml` y
`base_link`. Un geom que no se puede mover tiene la misma pose en el estimado
que en la verdad por construccion, asi que su fantasma no puede mostrar error.
**El indice de slot en la escena no es el id de geom del modelo**: quien mapee de
vuelta tiene que pasar por `ghost_geom_ids`.

Dos trampas ya pagadas, documentadas en `erp/viz/ghost.py`:

- `mjv_initGeom` deja `dataid = -1` en todo geom, y -1 en una malla no dibuja
  nada.
- El valor correcto es **`2 * geom_dataid`**, no `geom_dataid`: el contexto de
  render guarda **dos entradas por malla**. El valor sin duplicar dibuja la
  malla de otro eslabon en cada eslabon, y como cada malla trae su propio marco
  compilado, **se lee como un bug de rotacion**.

### Chequeo de arranque

```
fantasma sobre la planta al arrancar: 8.1e-08 m de separacion
```

`_check_ghost_starts_on_the_arm` compara `geom_xpos` de planta y fantasma antes
de abrir el lazo y aborta si se separan. Planta y filtro arrancan del mismo
equilibrio servoado, asi que tienen que coincidir. **Lo que NO cubre:** compara
poses, no que cada geom dibuje su malla — un `dataid` equivocado deja las poses
intactas. Eso lo cubre `test_viz_live.py` contra `mjv_updateScene`.

---

## 11. Teclado y graficos

### 11.1 El mapa, y por que estas teclas

```
  Q / A   rot_servo    +/-        P     fantasma on/off
  E / D   link1_servo  +/-        O     volver al reposo
  U / J   link2_servo  +/-        X     frenar el movimiento
                                  ESC   salir
```

**Las 26 letras ya estan tomadas** por el visor (`mjVISSTRING` y `mjRNDSTRING`).
No existe un juego de teclas que ignore; lo que se elige es *cual* colision:

| Clase | Ejemplos | Se puede deshacer? |
|---|---|---|
| banderas de `mjvOption` | Q camara, A auto-connect, E igualdad, D cuerpo estatico, U actuador, J junta, O objeto, P contacto partido, X textura | **Si** — viven en `viewer.opt`, que el handle expone |
| banderas de render | W wireframe, S sombra, R reflejo, G niebla, K skybox, L aditivo | **No** — viven en la `mjvScene` interna, que el handle pasivo no expone |

El mapa usa solo la primera clase, y `_restore_view_options` reescribe
`viewer.opt.flags` y `.geomgroup` desde una foto del arranque al principio de
cada tick, bajo `viewer.lock()`. Un toggle dura menos de un cuadro. Ademas se
pasa `show_left_ui=False, show_right_ui=False`.

**Las teclas quedan trabadas.** `key_callback` avisa cuando se APRIETA una tecla
y nunca cuando se suelta, asi que no hay evento de release con el cual parar: o
una tecla es un paso (0.01 rad, ~150 golpes para cruzar un rango) o queda
trabada y hace falta una tecla de freno. La demo traba; `X` limpia `held`, y `O`
tambien (si no, vuelve al reposo y se va de nuevo sola en el mismo tick).

Los rangos son **asimetricos** (`rot` ±2.79, `link1` 0..1.57, `link2` 0..1.04),
asi que un paso simetrico desde cero se sale del rango en dos de los tres en el
primer tick. Se recorta contra `model.actuator_ctrlrange`, leido del modelo.

La 4a junta (`act`) esta acoplada por tendon con rango `[0, 0]`: **sigue, no se
comanda**. Por eso "todas las juntas" son tres y no cuatro.

**Mando fino (2026-09-23).** Lo de arriba sigue valiendo para el VISOR. Tres
agregados, cada uno util solo:

- **Finura, teclas `1`..`5`**: (0.6, 0.3, 0.1, 0.03, 0.01) rad/s de jog y
  (5, 2, 0.5, 0.2, 0.1) grados de paso. El nivel 1 es la velocidad de antes.
  **`M`** alterna el modo paso a paso: cada tecla mueve un paso fijo y no
  traba nada. Los digitos no estan en `mjVISSTRING` ni `mjRNDSTRING`
  (verificado); la UI del visor los usa para `opt.geomgroup`, que el tick
  restaura. `M` es "centro de masa", bandera de `mjvOption`. La finura y el
  modo se escriben abajo a la izquierda de la escena con `viewer.set_texts`.
- **Panel de control** (`build_panel`, ventana Qt aparte, `--no-panel` lo
  saca). Ahi Qt SI entrega la suelta, asi que la junta se mueve **mientras**
  se mantiene la tecla o el boton -/+. El auto-repeat del sistema (pares
  suelta/aprieta con `isAutoRepeat`) se ignora; si el panel pierde el foco con
  una tecla apretada, se frena. Slider y caja numerica fijan un angulo exacto
  en grados, 0.1 de resolucion. Nunca toca `qpos`.
- **Objetivo vs consigna.** Teclas y panel mueven `target`; lo que va a los
  servos es `ctrl`, que se acerca a `target` a lo sumo `SLEW_RAD_S` = 1 rad/s
  (`erp.trajectory.slew_limit`), debajo de los ~2 rad/s donde saturan. Un
  angulo escrito lejos deja de ser un escalon, y `O` (reposo) tambien: antes
  era `ctrl[:] = q0`, un escalon. `X` pisa el objetivo con la consigna, para
  que la rampa no siga hacia un angulo escrito antes.

Verificado sin pantalla: 1 s de jog en finura 5 mueve 0.573 grados; un paso
en finura 3 es 0.500 grados; un objetivo de 2 rad en `rot` llega con la
consigna a 1.0000 rad/s como maximo y termina **exactamente** en 2.0, con la
planta en 2.000 y el estimado en 1.902 al llegar.

### 11.2 Strips

```python
plots = LivePlots(links=["link1", "link2"],
                  joint_labels=["rot", "link1", "link2"],
                  nis_target=12.0, ...)
acc = teleop.acc_cols   # [link1 xyz, link2 xyz] dentro de las 12 filas de IMU
```

**Tres ventanas, un grafico por variable** (2026-09-23). Una por link con un
grafico por eje de aceleracion (x, y, z), cada uno con planta (verde),
medicion aplicada (puntos ambar) y `h(x_hat)` (azul punteado); y una de juntas,
**un color por junta** -- planta continua, estimado punteado del mismo color --
sobre el NIS. La version anterior ponia
`link1_acc_x`, `link1_gyro_x` y las tres juntas en ejes compartidos, las juntas
todas del mismo verde, y no se distinguia que curva era cual. Tampoco
graficaba las mediciones ni el NIS: `push_measurement` y `push_nis` existian y
la demo nunca los llamaba. Ahora `Teleop.updates` guarda las actualizaciones
aplicadas en el tick, selladas en el instante de MUESTRA.

`acc_cols` se busca por nombre (`link1_acc`, `link2_acc`) dentro de
`imu_rows` y no se escribe `0:6`: depende del orden de bloques de
`cfg.imu.layout`, y con otro orden se graficaria el giroscopo en m/s² sin
ningun error. `nis_target` = 12 = cantidad de canales medidos.

**Ejes y fijos**: ±20 m/s² y −3..3 rad. Con autoescala el ruido de 0.05 m/s²
en reposo llena el grafico y parece que el brazo tiembla. Medido con un jog
que lleva cada junta de punta a punta: la planta queda dentro de ±15 m/s² y
`h(x_hat)` toca −30 en `link2_acc_x` un instante en una inversion; con prop,
los golpes llegan a ±60 y se recortan en el borde, donde igual se ven.

Redibuja cada 2 ticks (30 Hz) y no cada tick: matplotlib redibuja una figura
entera y no puede sostener eso, que es por que `[live]` es pyqtgraph y no
`[viz]`.

---

## 12. `--degrade`: que la demo PUEDA fallar

Una demo que solo se ve bien no distingue un stack que anda de uno que anda de
casualidad.

| Modo | Que rompe | NIS | Error de junta |
|---|---|---|---|
| (ninguno) | — | **5.70** | 0.003 rad |
| `swap` | la lectura de link2 entregada en la ranura de link1, con `rows` diciendo todavia link1. **Un solo lado** — permutar los dos es consistente y no degrada nada | **16 007** | 0.944 rad |
| `overconfident` | `R_declared = R / 100` | **589** | 0.007 rad |
| `zeros` | arranca del keyframe en vez del equilibrio servoado | **5.70** | — |

`swap` no cambia ni forma ni unidades: es el modo de falla silencioso de
ADR-0002 3.3.

`overconfident` es el caso didactico: **sigue el movimiento** y a la vez declara
una precision que no tiene. Por eso el diagnostico es el NIS y no el error.

`zeros` da un resultado bit-identico a sano, confirmando que el sesgo de 114
sigma del keyframe no sobrevive la recuperacion del tendon (~2 pasos).

---

## 13. Salida, y lo incomodo

Al cerrar:

```python
rep = consistency_report(np.asarray(teleop.nis), nz=12, warmup_fraction=0.5)
```

y despues, impreso a proposito en la salida y no solo en el mensaje de commit:

> el sensor virtual saca su ruido de **la misma `R`** que se le da al filtro y
> la planta **es** el modelo. Al filtro se le entrega exactamente el mundo que
> supone. Esto verifica la PLOMERIA — sellos de tiempo, filas, ejes, lazo — y
> **no** la calidad del estimador. Sobre el brazo real el mismo filtro da NIS
> mediana **29** contra el objetivo 12.

### Numeros de sesiones reales

| Corrida | Predicts | Updates | Descartadas | NIS mediana |
|---|---|---|---|---|
| jog guionado, 3 s | 1437 | 56 | 0 | 5.70 |
| sesion interactiva, ~40 s | 19 933 | 743 | **0** | 6.29 (media 15.75) |
| demo three-way (barrido seno) | — | — | 0 | 9.59 (media 11.12) |

El **0 descartadas** de la sesion interactiva es el numero que importa: es el
arreglo del horizonte aguantando bajo jogueo vivo y no sobre un guion de 3 s.

La brecha mediana/media es lo interesante y **no** aparece en la demo de
barrido: un jog de teclado es un comando **escalon** donde el seno es suave, asi
que cada apretada produce una rafaga de innovaciones grandes mientras un filtro
que no ve el comando lo infiere solo de las IMUs. Esa cola pesada es la tesis de
la demo apareciendo como aritmetica. **Leé la mediana para consistencia y la
brecha entre las dos para cuanto la estaban exigiendo.**

Headless corre a ~18x tiempo real, o sea ~6% de duty en tiempo real.

---

## 14. Correrlo y verificarlo

```bash
conda activate erp
python scripts/demo_teleop.py
python scripts/demo_teleop.py --degrade swap          # el fantasma se despega
python scripts/demo_teleop.py --degrade overconfident
python scripts/demo_teleop.py --no-plots              # solo el visor
```

**El lazo no tiene tests y es deliberado**: necesita pantalla y bloquea. Lo que
si esta cubierto en la suite rapida es la aritmetica de abajo:

| Test | Que fija |
|---|---|
| `test_live_sensor.py` | la grilla de muestreo y la latencia |
| `test_fusion_runner.py` | el runner, con una copia verbatim del viejo `run_imu_ekf` como oraculo de identidad bit a bit |
| `test_viz_live.py` | el buffer circular, y los geoms del fantasma contra `mjv_updateScene` |
| `test_sensor_contract.py` | el contrato del `Sensor`, una suite para todos |
| `test_ekf_linear.py` | el EKF contra un KF de solucion cerrada, sin mujoco |
| `test_plant.py` | el modelo ciego, el equilibrio, y que el escenario no mueve la fisica |

Antes y despues de tocar cualquier cosa de la ruta de estimacion:

```bash
python scripts/make_golden_run.py --check    # OK ... a rtol=1e-12
```

Si eso se pone rojo despues de un refactor, el refactor movio los numeros. **No
se regenera la fixture para que pase.**

---

## 14b. La variante sobre el brazo REAL: `scripts/demo_teleop_real.py`

Agregada 2026-09-23. **Todavia no se corrio sobre hardware**: todo lo de abajo
se verifico sin abrir COM6 ni COM7. La misma estructura que este demo -- sesion,
tick de Qt, visor con fantasma, `LivePlots`, panel -- y otras fuentes:

| Vista | Aca (sim) | En `demo_teleop_real.py` |
|---|---|---|
| planta | el modelo de planta, la verdad | el mismo modelo integrado bajo la **consigna** que se manda al brazo: un modelo de la consigna, no la verdad. `--plant-delay` la atrasa `arm_lag_s` = 0.395 s |
| fantasma | EKF sobre la IMU simulada | el EKF de las **celdas 18-19 del notebook** sobre las IMUs reales |
| medido | `LiveSimSensor` | `SerialIMUSensor`, TODAS las muestras, sesgo restado |
| graficos | acc | acc **y giro** por link (`LivePlots(kinds=...)`), ±20 m/s² y ±1.5 rad/s fijos |

Sin `--plant-delay` el fantasma va ~0.4 s atras de la planta en movimiento: es
el atraso de transporte del brazo, no el filtro.

**El filtro es el del notebook y no el de aca**: `P0` de la celda 18 (1/5/0.5
grados, no 1 grado plano) y el reposo calentado en el modelo de PLANTA. El
sesgo se estima en vivo con `rest_bias(expected_rest = h(x_rest))` sobre ~1 s
quieto despues del homing y se resta a cada `z` -- la aritmetica de `z -
imu_bias` de la celda 19. `apply_calibration` no se usa porque instalaria la
`R` de reposo que P6 rechazo. Una calibracion invalida aborta: viene con sesgo
cero y parece exito.

**Base de tiempo**: `perf_counter` absoluto, porque eso sella `SerialIMUSensor`
-- no el tiempo de simulacion de este demo. `buffer_horizon` = los 0.050 s del
config, redondeados a pasos enteros con `ceil(H/dt - 1e-9)` (25 pasos; sin el
epsilon, `0.05/0.002 = 25.000...04` daria 26).

**Seguridad** (decidido con el usuario): por defecto `--arm dry --imu sim`, sin
hardware; el brazo real pide `--arm real --arm-port` y escribir `si`. Homing a
[0,0,0] a velocidad 30, con la **llegada verificada leyendo los angulos**: las
tres juntas a 2 grados, 3 lecturas seguidas cada 0.2 s, o aborta a los 20 s.
`sync_send_angles` solo NO alcanza, y el notebook confia en el: en pymycobot
4.0.7, `MyPalletizer260.sync_send_angles` manda, sale del bucle en cuanto
`is_moving()` da 0 y **devuelve 1 siempre**, incluso por timeout -- con ~0.4 s
de cola en el firmware, el primer `is_moving()` puede llegar antes de que el
brazo arranque. Verificado sin hardware con un transporte que imita esa vuelta
inmediata: espera 15 lecturas lejanas y 3 buenas; aborta si nunca llega, si
`get_angles` da `-1`, o a 2.5 grados; una lectura buena suelta en movimiento no
cuenta. Segunda red: si el brazo se mueve en la ventana de reposo, el
giroscopo invalida el sesgo y tambien aborta. Tope de consigna 0.5 rad/s (y las
velocidades de jog recortadas a el); `ctrlrange` verificado al arrancar dentro
de los limites de la API. 25 Hz de consignas en promedio, solo si cambian. En el
visor las teclas de junta dan UN paso con el brazo real; continuo solo desde el
panel. Frenar, cerrar, Ctrl-C o una IMU muerta cortan las consignas; los servos
NO se liberan. El freno de emergencia es el interruptor.

**Verificado sin hardware** (ensayo con reloj virtual; la ruta "brazo real"
contra el `FakeMyCobot` de `test_robot.py`, la IMU real contra el `FakePort` de
`conftest.py`):

- 354 consignas en 18 s de manejo: todas dentro de `ctrlrange` y de la API, J4
  en 0, `send_radians` a velocidad 100; tasa de consigna maxima **0.500 rad/s**.
- Envios: **25.0 Hz en promedio**, con intervalos que alternan **33 y 50 ms** --
  la grilla absoluta de 40 ms cae en ticks de 60 Hz. Por eso "a lo sumo 25 Hz"
  vale en promedio, no intervalo por intervalo.
- Al soltar, el objetivo queda fijo, la consigna lo alcanza y **no se manda
  nada mas**. Una tecla del visor con el brazo real da un paso y no traba.
- La guarda salta ante un salto de consigna; despues de una falla no sale nada.
- IMU real (falsa): sesgo inyectado recuperado a **4e-7**, 0 descartadas con
  sellos de llegada, y el lector muerto -> `SensorError` -> falla -> 0 envios.
- EKF sobre la IMU simulada: 0 descartadas, NIS mediana **6.9**. Error de junta
  tras 18 s de movimiento: rot **0.019**, link1 **0.008**, link2 **0.003** rad.
  rot es el peor porque gira alrededor de la vertical: la gravedad no lo ve y
  solo el giroscopo lo observa, asi que deriva. Esperalo tambien en el brazo.
- `--plant-delay` agrega **395.0 ms** sobre los ~46 ms del servo simulado. La
  primera version sostenia la historia de consignas (un punto por tick) y
  agregaba 411 ms; ahora se interpola a 500 Hz.
- Rangos bajo el tope: planta acc -9.8..11.2 m/s², giro ±1.0 rad/s; estimado
  acc -9.7..12.2, giro ±0.6. De ahi ±1.5 rad/s para el giro.

**La primera vez sobre hardware**, en este orden: `--imu real` solo (~20 Hz,
sesgo valido, `descartadas` 0, NIS finita); despues `--arm real --imu real`,
una junta, finura 5 en pasos, con la mano en el interruptor. Cada sesion se
guarda en `data/sessions/<fecha>/` (en `.gitignore`), con el CSV **crudo**.

---

## 15. Referencias

- `docs/adr/0002-notebook-to-package.md` — el plan del que salio todo esto
- `.claude/skills/goal/references/teleop.md` — mecanica verificada de la API y
  las trampas; `references/invariants.md` — que no se puede romper
- `CLAUDE.md` — convenciones, hechos de hardware medidos, estado del repo
