# Handoff: envolvente J2 → J3 del MyPalletizer260 (medición del 2026-10-21)

> **Actualización, mismo día, más tarde: pasos 1–3 de la sección 6 hechos.**
> El resto del documento es el registro de antes del arreglo y se deja como
> estaba; donde dice "no se hizo" o nombra los archivos con la etiqueta
> invertida, es historia.
>
> - **Paso 1.** `MEDIR_MAX` y `MEDIR_MIN` quedaron en `False`. La celda de
>   `send_angles` ahora está detrás de `MOVER_A_CERO = False`: correr el
>   notebook de arriba abajo ya no mueve ni libera el brazo.
> - **Paso 2.** Se renombraron las dos grabaciones (`max`↔`min`):
>   `j2j3_min_20261021_121758.csv` es ahora el tope inferior y
>   `j2j3_max_20261021_121951.csv` el superior. Se volvió a correr el ajuste,
>   que pisó `data/processed/j2j3_envelope.csv`. El CSV nuevo reproduce la
>   tabla de la sección 3 en todos sus puntos y las fórmulas compactas. La
>   sección 7 usa los nombres viejos: con los nuevos, `inf` es
>   `j2j3_min_..._121758` y `sup` es `j2j3_max_..._121951`.
> - **Paso 3.** El ajuste ya no exporta si: la mediana de J3 de un archivo
>   cae del lado del otro tope (`J3_ENTRE_TOPES_DEG = 0`), algún `max` no
>   supera a todo `min`, J3_min ≥ J3_max sin margen, o un control da
>   "0 de 0". En esos casos la celda termina con error. Corrido contra los
>   archivos con los nombres viejos, los cuatro saltan y no se escribe nada.
>   Al grabar, un recorrido con la mediana del lado equivocado se guarda como
>   `j2j3_revisar_{lado}_…csv`, que el ajuste no lee. El protocolo habla del
>   signo de J3 en la API, no de "arriba/abajo". La prueba sin brazo tiene
>   dos falsificaciones nuevas para esto.
> - **Paso 5, la parte de código, hecho (sin el brazo).** La guarda existe:
>   `erp.robot.CoupledLimit` lee este CSV, `JointMap.check_setpoint` lo
>   aplica, y el `send` de `DryRunArm` y de `MyPalletizerArm` lo corre antes
>   de tocar el transporte. `scripts/demo_teleop_real.py` recorta el objetivo
>   y la consigna a la envolvente. `MyPalletizerArm` rechaza, antes de abrir
>   el puerto, una envolvente con menos de dos grabaciones por lado. El
>   notebook `mypalletizer260EKF.ipynb` todavía no la carga.
> - **Paso 4, segunda grabación por lado (12:56 y 12:58).** Las dos salieron
>   **otra vez con la etiqueta invertida**: la del recorrido `max` tenía J3
>   con mediana −66.8° y la del `min` +51.3°. El chequeo nuevo las guardó como
>   `j2j3_revisar_*` y el ajuste no las leyó. Son cuatro de cuatro: el lado
>   del brazo que se siente "arriba" es J3 **negativo** en la API. Se
>   renombraron al tope que contienen (`j2j3_min_20261021_125636.csv`,
>   `j2j3_max_20261021_125836.csv`), se volvió a correr el ajuste y se pisó el
>   CSV:
>   - **Tope superior:** las dos grabaciones coinciden en 1.25° (en
>     J2 = 86.2), así que el margen es 1.25°. La zona plana da 63.1° en las
>     dos, o sea que el valor forzado se repite. En la práctica no manda: la
>     caja (J3 ≤ 60) y el `ctrlrange` (59.6) quedan por debajo de
>     `j3_max_seguro` = 61.85.
>   - **Tope inferior:** difieren hasta **7.70°** (en J2 = 26.2), y ese es el
>     margen. Es grande, pero ahora está medido.
>   - **Control:** 31 de 1182 muestras de la grabación vieja del 24/09 quedan
>     fuera de envolvente + margen, con J2 entre 48.2 y 49.9.
>
>   Con dos grabaciones por lado la envolvente queda **lista**, y
>   `demo_teleop_real.py --arm real` ya no la rechaza. Límite superior con
>   margen: 57.1° en J2 = 45, 44.8° en J2 = 60, 30.8° en J2 = 75 y 15.0° en
>   J2 = 90.
> - **Sigue abierto:** llevar la envolvente al `JMAP` del notebook
>   `mypalletizer260EKF.ipynb`. Si el margen de 7.7° del tope inferior parece
>   demasiado, una tercera grabación de ese lado lo afinaría. En la zona
>   plana el CSV dice 63.1° (`j3_max_seguro` 62.1°), que es el valor
>   **forzado** de la sección 4, no los ~58° de contacto suave. El margen es
>   el mínimo de 1° porque hay una sola grabación por lado. **Todavía no
>   sirve como guarda de comandos.**

**Estado:** se hizo una pasada por lado (`max` y `min`). **Las etiquetas de los
dos archivos quedaron invertidas**, así que el `data/processed/j2j3_envelope.csv`
que exportó el notebook **no sirve** y no hay que usarlo. Los datos crudos sí
sirven: con las etiquetas intercambiadas dan una envolvente coherente, en la
tabla de abajo. Hay dos cosas abiertas: el tope superior **cede ~4° al empujar
fuerte**, y no hay repetición de ningún lado, así que el margen de seguridad
todavía no está medido. No se commiteó nada.

Notebook: `notebooks/Palletizer.ipynb`, celdas 3–9 (el markdown de la celda 3
explica cómo mide). Este documento es el registro de lo medido; el notebook es
el procedimiento.

## 1. Qué se midió

| Archivo (`data/raw/`) | Etiqueta en el nombre | Qué contiene en realidad | n | Duración |
|---|---|---|---|---|
| `j2j3_max_20261021_121758.csv` | `max` | **tope INFERIOR** de J3 | 1348 | 89.8 s |
| `j2j3_min_20261021_121951.csv` | `min` | **tope SUPERIOR** de J3 | 1348 | 89.8 s |

- Servos liberados y brazo movido a mano. Lectura por `get_angles()` en grados
  API: dt mediano 0.067 s (~15 Hz), 1 lectura fallida por archivo y resolución
  de 0.08°.
- Cobertura de J2 completa en los dos archivos: 37/37 tramos de 2.5° en
  −2..90°. J2 recorrió −7.2..94.1°.
- J1 se movió 155° en el archivo del tope superior. Es irrelevante para el
  acople J2–J3.
- Siguen en el repo, y se usan como control, dos grabaciones viejas mezcladas
  (tope inferior, superior y tránsito en un mismo archivo):
  `j2j3_envelope_raw_20260924_172730.csv` y
  `j2j3_envelope_raw_20261021_115514.csv`. `j2j3_envelope_raw.csv` está
  excluida: 589 de sus 598 muestras tienen J2 > 90.

## 2. Problema 1: etiquetas invertidas

**Evidencia.**
- El archivo `max` llega a J3 = −106.3° y su tramo J2 ∈ [88, 95) tiene
  mediana −105.0°: es el tope inferior.
- El archivo `min` nunca baja de J3 = 2.2° y en la zona plana ronda 58–63°:
  es el tope superior.
- La salida de la celda de ajuste da `J3_min ≈ 58` > `J3_max ≈ −50` en todos
  los tramos y avisa que *"con margen el intervalo se cierra"* en todo el
  rango. El chequeo contra las grabaciones viejas dice *"0 de 0"* porque, con
  el intervalo cerrado, `limites_j3` devuelve NaN en todos lados: el chequeo no
  falló, no tuvo con qué comparar.

**Consecuencia.** El ajuste usó el percentil del lado equivocado en cada
archivo, y se quedó justamente con las muestras en que la mano se había
soltado del tope. El CSV exportado es basura.

**Causa probable.** No está confirmada. El protocolo decía *"llevá J3 arriba
hasta el tope"* para `max`, pero "arriba" en el brazo físico puede
corresponder a J3 **negativo** en la API. La otra posibilidad es que se hayan
corrido las celdas en el orden cambiado. Cualquiera de las dos se evita
redactando el protocolo en términos de la API ("J3 hacia valores positivos:
en la línea en vivo J3 sube") y validándolo en el ajuste (paso 3 de la
sección 6).

**Arreglo de los datos** (no se hizo; es una decisión del dueño de los datos):
renombrar los dos archivos intercambiando `max`↔`min`, volver a correr la celda
de ajuste y pisar el CSV exportado. Hasta entonces, la sección 3 tiene los
números correctos, calculados en memoria y sin tocar archivos.

## 3. Resultado con las etiquetas corregidas

Mismo método que el notebook: p90 en el tope superior y p10 en el inferior,
tramos de 2.5°, cada punto con el J2 de su propia muestra e interpolación
lineal. Grados API, **sin margen**:

| J2 | J3 mín (tope inf.) | J3 máx (tope sup.) | Rango de J3 |
|---:|---:|---:|---:|
| −6.25 | −13.1 | 63.1 | 76.2 |
| 3.75 | −23.5 | 63.1 | 86.6 |
| 13.75 | −34.8 | 63.1 | 97.9 |
| 23.75 | −45.6 | 63.1 | 108.7 |
| 33.75 | −55.0 | 63.1 | 118.1 |
| 43.75 | −65.2 | 60.0 | 125.1 |
| 53.75 | −74.3 | 51.7 | 126.1 |
| 63.75 | −84.0 | 42.9 | 126.9 |
| 73.75 | −90.8 | 33.3 | 124.1 |
| 83.75 | −101.9 | 23.3 | 125.2 |
| 91.25 | −106.1 | 14.3 | 120.5 |

Con J2 ≥ 93.75 no hay valor: quedó fuera de lo medido y el método no
extrapola.

**Fórmulas compactas** (solo informativas; la tabla es la salida):
- Tope inferior: `J3_min ≈ −0.957·J2 − 21.9`, residuo máximo 5.2°. Casi
  recto, pero no alcanza como fórmula.
- Tope superior: plano en 63.2° hasta J2 ≈ 40.9°, y después
  `63.2 − 0.932·(J2 − 40.9)`, residuo máximo 2.1°. El valor plano está inflado
  (ver la sección 4).

**Qué límite manda, comparado con las cajas fijas que usa hoy el repo**
(`JointMap.api_limits_deg`: J3 −92..60):
- **Tope superior:** para **J2 ≳ 45°** la curva está por debajo de 60° y es
  la que manda. Con J2 = 90° el máximo real es ~15°, no 60°.
- **Tope inferior:** manda la curva para **J2 ≲ 72°**. Más allá, el tope
  mecánico baja de −92° y manda el límite de la API.
- **Riesgo concreto:** `scripts/demo_teleop_real.py` deja comandar link2 = J3
  hasta 59.6° (su `ctrlrange`) con cualquier J2. Para J2 ≳ 45° eso empuja J3
  contra el tope mecánico.

## 4. Problema 2: el tope superior cede al empujar fuerte

En la zona plana (J2 entre −2° y 38°), el tope superior es **bimodal**:

| J3 | Muestras | Cuándo |
|---|---:|---|
| 58.0–60.0° | 98 | hasta t ≈ 34 s |
| 62.5–63.5° | 165, de las cuales 149 exactamente en ~63.1° | desde t = 34.3 s hasta el final |

- Las grabaciones viejas nunca pasan de 57.2° (la del 24/09) y 58.35° (la de
  hoy, antes de este recorrido).
- **El salto aparece solo en la zona plana.** Comparando cada tramo antes y
  después de t = 34.3 s: +3.6 a +4.4° para J2 < 40°, y entre −0.7° y +1.6°
  en la zona con pendiente.
- Un corrimiento de la lectura de J3 movería todo por igual, así que no es
  eso. Lo más probable es que el tope de la zona plana sea blando: ~58–59.5°
  con contacto suave y ~63.1° empujando fuerte. No se verificó mecánicamente.

**Por qué importa.** El método supone que las desviaciones respecto del tope
son siempre hacia adentro (la mano que afloja). Un tope que cede rompe ese
supuesto: empujar más fuerte da valores **hacia afuera**, y el p90 elige
justamente esos. Por eso el 63.1° de la tabla es el valor "a la fuerza", no
uno seguro para comandar. Para comandar el brazo, en la zona plana conviene
usar ~58°, el valor de contacto suave que coincide con las grabaciones viejas.

**El tope inferior también depende de cómo se lo empuje.**
- La segunda mitad del recorrido lee de 1 a 10° más abajo que la primera.
  Probablemente en la primera pasada no se apoyó bien: hay tramos con 3–6
  muestras.
- La grabación vieja del 24/09 coincide con este ajuste dentro de ~1° en
  J2 ≈ 35–40, 55–60 y 85–90°. Difiere en 4–6.6° en J2 ≈ −5–0, 25–30 y 45–50°,
  y en esos tramos este ajuste da **más bajo**.
- Contra este ajuste, la grabación vieja del 24/09 tiene 180 muestras más de
  1° por debajo del tope inferior (peor caso 4.85° debajo). La de hoy tiene 3
  (peor caso 4.35° debajo). Cuando dos grabaciones difieren así en un mismo
  tramo, en alguna de las dos la mano no llegó al tope.

## 5. Qué es confiable

| Afirmación | Confianza |
|---|---|
| El tope superior **cae** con J2 a partir de J2 ≈ 41–45°, hasta ~14° en J2 = 91° | **Alta.** Repetible antes y después del salto (±0.7°, +1.6° en un tramo) |
| El tope inferior baja casi lineal, ~−0.96°/° de J2, hasta ~−106° | **Alta en la forma.** El nivel tiene ±5° según el tramo |
| Valor del tope superior en la zona plana | **Bajo.** 58–59.5° suave vs 63.1° forzado; usar ~58 |
| Margen de seguridad | **No medido.** Hay una sola grabación por lado; el notebook usó el mínimo de 1° |
| `data/processed/j2j3_envelope.csv` | **Inválido.** Etiquetas invertidas |

## 6. Próximos pasos, en orden

1. **Antes de abrir el notebook otra vez:** las celdas 7 y 8 quedaron con
   `MEDIR_MAX = True` y `MEDIR_MIN = True`. Correr el notebook de arriba
   abajo libera los servos. Volver a ponerlas en `False`.
2. **Arreglar las etiquetas** (renombrar `max`↔`min`), volver a correr el
   ajuste y pisar el CSV exportado. Otra opción es borrar el CSV hasta tener
   datos buenos.
3. **Validar las etiquetas en el ajuste.** Si la mediana de J3 del archivo
   `max` es menor que la del `min`, abortar con un mensaje en vez de exportar.
   Habría frenado este error. En el protocolo, escribir "J3 hacia valores
   positivos / negativos de la API" en lugar de "arriba / abajo".
4. **Repetir cada lado al menos una vez más, con fuerza de contacto
   constante y suave**, para medir la dispersión, que es el margen de verdad.
   En el tope superior, decidir explícitamente si vale el contacto suave
   (~58°) o el forzado (~63°). Para una guarda de comandos, el suave.
5. **Recién con eso,** llevar la tabla a la guarda de `demo_teleop_real.py`
   (y a `JointMap.validate`) para que no comande J3 > f_max(J2). Por ahora se
   decidió dejarlo solo en el notebook.

## 7. Cómo reproducir los números de la sección 3 sin tocar archivos

Corriendo antes las celdas de configuración y de funciones del notebook (4 y 5):

```python
g = lambda n: np.loadtxt(DIR_RAW / n, delimiter=",", skiprows=1)
inf = g("j2j3_max_20261021_121758.csv")   # etiqueta 'max', contiene el tope INFERIOR
sup = g("j2j3_min_20261021_121951.csv")   # etiqueta 'min', contiene el tope SUPERIOR
envs = {"min": ajustar_lado(inf[:, 2], inf[:, 3], "min"),
        "max": ajustar_lado(sup[:, 2], sup[:, 3], "max")}
lo, hi = limites_j3(np.arange(-6.25, 95, 2.5), envs)
```
