# Cambios del FSR y la cúpula (2026-11-15 a 2026-11-17)

Este documento resume lo que se cambió en el repo durante el trabajo sobre el
FSR de la punta: qué se hizo, por qué, qué se midió y qué quedó pendiente.

- Rama `feat/ekf-fsr`, sobre `a4cd8cb`. **Nada está commiteado todavía.**
- Todas las mediciones se hicieron en la **máquina A** (env `erp`).

## Resumen

| # | Qué | Archivos | Estado |
|---|---|---|---|
| 1 | CLAUDE.md al día con lo medido | `CLAUDE.md` | hecho |
| 2 | El FSR entra al filtro como una fila de la medición | `estimators/ekf_fsr.py`, `estimators/__init__.py` | hecho, con tests |
| 3 | Verlo en vivo: ventana de contacto y cúpula coloreada | `viz/live.py`, `viz/strips.py`, `scripts/demo_teleop.py` | hecho; la ventana no se vio en pantalla |
| 4 | `medir_penetracion.py` por la misma ruta que el demo | `scripts/medir_penetracion.py` | hecho, re-medido |
| 5 | Script que mide la silicona desde el ensayo de banco | `scripts/ajustar_cupula.py` (nuevo) | **a medio hacer** |
| — | Handoff de cuerpo deformable (`flexcomp`) | — | **evaluado y descartado** |

**Verificación al 2026-11-17:**

| Chequeo | Resultado |
|---|---|
| `pytest -q` | **413 passed, 1 skipped** (eran 403 en `a4cd8cb`) |
| `pytest -q -m "not mujoco"` | 341 passed |
| `mypy software/src` | limpio |
| Regla de imports (grep) | no imprime nada |
| `scripts/make_golden_run.py --check` | `OK ... a rtol=1e-12` (el camino de las IMUs no cambió ni un dígito) |
| `ruff` sobre el paquete y los tests | sigue rojo solo por `software/tests/test_EKF.py`, el archivo de prueba suelto que ya estaba |

`ruff` también marca líneas largas en `scripts/ajustar_cupula.py`, que está a
medio hacer.

---

## 1. CLAUDE.md al día

Se actualizaron las partes que habían quedado viejas.

- **Smart App Control:** ahora `import mujoco` funciona en la máquina A. Se
  dejó el diagnóstico por si vuelve.
- **Conteos de tests:** 403 pasó a 413, y 313 (rápidos) pasó a 341.
- **Clase de contacto:** el adaptador del FSR ya no usa su `solref` propio;
  usa la clase `herramienta` de `contacto.xml`.
- **Commit `a4cd8cb`:** se agregó la entrada que faltaba.
- **Arquitectura, ítem 7:** se agregó un párrafo nuevo, "el FSR llega como una
  fila de la medición" (ver la sección 2).

## 2. El FSR entra al filtro (`software/src/erp/estimators/ekf_fsr.py`)

**Antes**, `EKFFSR.update_contacto(bool)` existía, pero solo
`medir_penetracion.py` lo llamaba. Lo hacía por fuera del `FilterRunner`, en
cada cuadro del render y sin respetar el instante de la muestra.

**Ahora** la fuerza del FSR viaja como **una fila más de la `Measurement`**:
- la fila `fsr_force` (el `<touch>` de la cúpula), en newtons;
- es la fila 15 de `MyPalletizer260_fsr.xml`, igual en la planta y en el
  modelo del filtro.

`EKFFSR.update` separa esa fila:
1. Las demás filas (las IMUs) van al `EKF.update` de siempre.
2. La fila del FSR va a `update_contacto(F > umbral_n)`: ON si supera 1 N.

Qué gana esto:
- `FilterRunner`, `Measurement` y `Sensor` **no cambiaron**.
- El contacto se aplica en el paso de **su** muestra, con el mismo buffer y
  orden que las IMUs.

Lo nuevo en el archivo:
- **Parámetros:** `fila_fsr` (índice de la fila; `None` = comportamiento de
  siempre) y `umbral_n` (1 N).
- **`Contacto`:** el registro de cada lectura, con paso, fuerza, ON/OFF,
  distancia antes de corregir, innovación y NIS. Se vacía con
  `tomar_contactos()`.
- **`predict`** cuenta los pasos, para poder sellar cada `Contacto` en el tiempo.
- **El NIS que devuelve `update` es solo el de las IMUs**, para no mezclar
  magnitudes distintas en el historial.
- **Sigue ciego:** no hay `u` en ninguna firma.

**Regla importante:** esa fila **nunca** puede llegar al `EKF.update` común.
El modelo del filtro no tiene contacto, su `<touch>` siempre lee 0, y la
fuerza entera se volvería innovación de un `h` que no la puede explicar. Hay
un test que lo falsea.

**Tests nuevos** (`software/tests/test_ekf_fsr.py`, 6 casos, todos rápidos):
- con la fila del FSR, `update` da **bit a bit** lo mismo que IMU + contacto;
- **falsificación:** la fila como canal común da un NIS de más de 1000×, y el
  contacto no se aplica;
- sin la fila, el resultado es idéntico a `EKF`;
- cada lectura queda registrada con su paso;
- una medición con solo el FSR reporta el contacto;
- **a través del `FilterRunner`**, el contacto cae en el paso de su sello (5).
  La ruta vieja, "al tiempo actual del filtro", caía 12 pasos después.

## 3. Verlo en vivo

```
python scripts/demo_teleop.py --prop fsr               # la punta del FSR + ventana de contacto
python scripts/demo_teleop.py --prop fsr_escaleras     # lo mismo, con las escaleras reales
python scripts/demo_teleop.py --prop fsr --sin-fsr     # falsificación: mapa sin FSR
```

**Hallazgo:** hasta ahora `--prop fsr` cargaba la punta en la planta y nada
más. El filtro era un `EKF` común y el FSR no llegaba a ningún lado.

Qué cambió en `scripts/demo_teleop.py`:
- **Prop nueva** `fsr_escaleras`, y la lista `FSR_PROPS`.
- **Con FSR, el filtro corre sobre el brazo con adaptador**: es el único
  modelo con la fila 15, y no tiene masa, así que su dinámica es idéntica.
- **El mapa de sólidos** para el contacto es la planta misma.
- **`LiveSimSensor`** ahora muestrea las 12 filas de las IMUs más la del FSR,
  en la misma muestra.
- **Constantes, definidas una sola vez:**
  - `FSR_UMBRAL_N = 1.0`;
  - `PEN_ON_M = 0.0004`;
  - `SIG_FSR_N = 0.2`: ruido del FSR virtual, nominal. Chico a propósito, para
    no inventar toques;
  - `DESTELLO_S = 0.3`.
- **`--sin-fsr`:** el filtro tiene el mapa, pero nadie le da el FSR. Es la
  falsificación: el estimado entra al sólido y nadie lo saca.

Lo que se ve:
- **Una cuarta ventana**, "FSR / contacto", en `software/src/erp/viz/live.py`
  con `LivePlots(contact=...)`:
  - arriba, la fuerza del FSR: la de la planta, la medida y el umbral;
  - abajo, la distancia con signo de la cúpula al sólido, planta contra
    estimado, con cada corrección marcada (triángulo verde = ON, naranja = OFF
    que saca del sólido);
  - la distancia se recorta en +40 mm, así que una línea pegada arriba quiere
    decir "lejos".
- **La cúpula del fantasma pintada** en el visor:
  - verde, una corrección ON reciente;
  - naranja, una OFF que lo sacó del sólido;
  - rojo, adentro sin corrección.
- **Texto arriba a la derecha:** la fuerza medida, ON/OFF, las distancias y
  los conteos.
- **Los buffers de la ventana** están en `ContactTrace`
  (`software/src/erp/viz/strips.py`). Es solo numpy y tiene 2 tests en
  `test_viz_live.py`. `viz/live.py` sigue sin importar mujoco; la esfera del
  visor la dibuja el script.

**Verificado sin pantalla:** las tres configuraciones arrancan con la fila 15,
cero descartes y el fantasma a 6e-8 m de la planta. La ventana en sí **no se
miró**.

## 4. `medir_penetracion.py` por la misma ruta

Ahora pasa el FSR igual que el demo: sensor → `FilterRunner` → `EKFFSR.update`.
Toma las constantes de `demo_teleop.py` en vez de repetirlas.

**Lo medido** (escenario piso; el docstring de `correr` lo guarda):

| Cómo llega el FSR | Penetración del estimado | Recuperación | ON/OFF |
|---|---|---|---|
| Sin FSR | 22.27 mm | 0.13 s | — |
| Ruta vieja: lectura directa a 60 Hz, sin ruido | 5.64 mm | 0.02 s | 159 / 4 |
| **Ruta nueva: 20 Hz, en la línea de las IMUs** | **10.97 mm** | 0.05 s | 50 / 3 |
| Ruta nueva sin ruido | 10.97 mm (idéntico) | 0.05 s | 50 / 3 |
| Ruta nueva a 60 Hz (subieron también las IMUs) | 0.80 mm | 0.00 s | 150 / 5 |

**Conclusión:** importa la **frecuencia** del FSR, no su ruido. A 20 Hz el FSR
saca la mitad de la penetración; a 60 Hz casi toda. La ruta vieja era
optimista: leía la planta a la tasa del render, cosa que el hardware no puede
dar.

## 5. `scripts/ajustar_cupula.py` (nuevo, a medio hacer)

**Objetivo:** medir la silicona en el ensayo de banco
(`electronics/FSR/ensayo-fsr-20261109-1931.csv`, todavía sin trackear) y
proponer la blandura del contacto (`solref`/`solimp`) para que la cúpula
simulada se hunda como la real.

**Por qué hace falta:** hoy la cúpula simulada es ~8 veces más rígida que la
real. En simulación se hunde 0.44 mm con 24–32 N; en el ensayo, ~2.6 mm con
~15 N.

**Lo que hace hoy:**
- **Limpieza:** el cero de cada celda, por ciclo.
- **Diagnóstico del banco:** tasa real de las celdas, atraso y lazo invertido.
- **Ajuste de Hertz sobre la rama de carga:** ahí todos los defectos del
  banco se reducen a un corrimiento constante de z.
- **Curva contra un plano**, que es como apoya la cúpula en el robot.
- **Calibración del FSR**, de conductancia a fuerza.
- **`--autoprueba`:** datos sintéticos con la verdad conocida. Recupera E
  (+3.1 %), el exponente (1.506) y el toque aparente.

**Lo que falta:**
- **El paso 3 (por ciclo) se cae con los datos reales.** La edición que lo
  arreglaba fue rechazada y no se reintentó.
- **Una falsificación que de verdad falle.** La actual, el ajuste ingenuo,
  solo erra −2.5 %.
- **El ajuste de dos regímenes** (ver los hallazgos).
- **El paso 6:** buscar en MuJoCo el `solref`/`solimp` que reproduce la
  curva. Hoy es solo un esqueleto.
- Las líneas largas que marca `ruff`.

## Hallazgos del ensayo de banco

Medido con `ajustar_cupula.py` y análisis sueltos sobre el CSV:

| Qué | Valor |
|---|---|
| Celdas de carga | una muestra nueva cada **100 ms (10 Hz)**, no 100 Hz |
| FSR y z | se registran en cada fila (50 Hz) |
| Atraso de la fuerza con z quieto | **0.19 s** en las celdas (p10–p90: 0.17–0.22); 0.03 s en el FSR |
| Lazo fuerza–z crudo | **invertido en el 100 % de los ciclos**, y el del FSR también: hay juego en z, que con este ensayo no se separa de la pérdida del material |
| Antes del contacto | un bulto espurio de hasta **0.48 N**, con el FSR plano. Es del banco, no de la cúpula |
| Toque aparente en la carga | z ≈ +0.3 a +0.5 mm, coherente con juego/2 + velocidad × atraso |
| FSR | no reacciona hasta **~2 N** y satura cerca de 15 N |
| Forma de la curva en la carga | Hertz (∝ δ^1.5) hasta ~1 mm y **lineal** después (exponente ≈ 1.06) |

La transición a lineal es lo que hace una punta esférica **chica** cuando
queda enterrada: el área de contacto deja de crecer y se comporta como un
punzón plano (Boussinesq, F = 2·a·E*·δ). Eso pide ajustar en dos regímenes:
Hertz al principio y Boussinesq después. Las dos estimaciones de E tienen que
coincidir, y eso además valida el radio del punzón.

Implicancia para el filtro: **el umbral de 1 N del FSR es marginal en el
sensor real**, que no reacciona hasta ~2 N.

## Lo que se descartó: el cuerpo deformable con `flexcomp`

Se evaluó un handoff que proponía modelar la cúpula como silicona deformable
(`<flexcomp>` con malla tetraédrica) y leer el FSR como deformación entre dos
nodos. **No se adoptó**, por cinco razones:
1. **Rompe el filtro.** Ese XML también es el modelo del filtro: el estado
   pasaría de 11 a cientos de variables, y el filtro se vuelve inviable.
2. **El callback de sensor de MuJoCo es global al proceso.** Correría también
   en el modelo del filtro y en sus diferencias finitas.
3. **Habría que cambiar el paso de simulación.** La silicona pediría ~0.1 ms
   contra los 2 ms actuales, y eso invalida el ajuste del contacto.
4. **Mide la variable equivocada.** Un FSR mide fuerza, no deformación entre
   dos nodos, y esa fuerza ya la da el `<touch>`.
5. Mete un voltaje en `sensordata` y no dice cómo lo usa el filtro.

La silicona se representa con la blandura del contacto, y la sección 5 es la
forma de medirla bien.

## Pendientes y decisiones abiertas

1. **El radio de la punta del punzón.** ¿Es una semiesfera en la punta de una
   varilla? ¿De qué diámetro? Todo E depende de ese dato.
2. **Seguir o no con `ajustar_cupula.py`:**
   - el arreglo del paso 3;
   - el ajuste de dos regímenes;
   - una falsificación real;
   - el paso 6 en MuJoCo.
3. **La frecuencia del FSR en el firmware.** Hace falta un barrido en
   simulación (IMUs fijas a 20 Hz y FSR aparte a 20, 50, 100 y 200 Hz) para
   saber qué pedir. `EKFFSR` ya acepta mediciones que traen solo el FSR.
4. **Fase 2 del plan del FSR, sin empezar:** leer el FSR real.
   - una entrada en `config/estimation.yaml`;
   - una columna nueva en el decoder;
   - `demo_teleop_real.py` con `EKFFSR`;
   - la tabla de calibración, que sale del ensayo.
5. **Commit.** Nada está commiteado. `software/tests/test_EKF.py` sigue
   rompiendo `ruff` (y por lo tanto CI); hay que borrarlo o convertirlo en un
   test.

## Cómo verificar

```bash
PY=C:\Users\nicoa\anaconda3\envs\erp\python.exe
$PY -m pytest -q                                      # 413 passed, 1 skipped
$PY -m mypy software/src                              # limpio
$PY scripts/make_golden_run.py --check                # OK a rtol=1e-12
$PY scripts/medir_penetracion.py --escenario piso --filtro-modela nada fsr
$PY scripts/demo_teleop.py --prop fsr                 # mirar la ventana de contacto
$PY scripts/ajustar_cupula.py --autoprueba            # recupera E; la falsificación todavía es floja
```
