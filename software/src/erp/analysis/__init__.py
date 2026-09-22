"""Consumidores de solo lectura de una corrida terminada.

Fase P8 de ADR-0002, y la capa 4 de 4.2: nada de aca participa del filtrado, se
limita a medir lo que el filtro ya produjo. Por eso puede importar ``sim/`` y
``fusion/``, y nada dentro del stack de estimacion puede importarla a ella.

**Castellano**, por la razon que fija 5.3: ``estimate_lag`` y
``propagate_to_site`` se movieron desde ``scripts/make_golden_run.py`` sin
tocarles el cuerpo NI el docstring, que estaban en castellano. Un paquete mitad
y mitad seria peor que cualquiera de las dos opciones enteras, asi que el modulo
nuevo (``consistency.py``) sigue a los que se mudaron. ``erp/viz/``, en cambio,
quedo en ingles, porque su ``__init__.py`` ya lo estaba.

Importar este paquete trae mujoco, porque ``propagate_to_site`` evalua h y H. No
carga ningun modelo -- eso lo hace el llamador -- pero es lo que obliga a marcar
con ``mujoco`` cualquier test que lo ejercite de punta a punta.
"""

from erp.analysis.consistency import ConsistencyReport, consistency_report
from erp.analysis.lag import estimate_lag
from erp.analysis.site import propagate_to_site

__all__ = [
    "ConsistencyReport",
    "consistency_report",
    "estimate_lag",
    "propagate_to_site",
]
