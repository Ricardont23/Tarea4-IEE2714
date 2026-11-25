# Tarea4-IEE2714
Repositorio de códigos e imágenes (y demás) usados para el desarrollo de la tarea 4.  

## Pregunta 1

> La imagen `belgium.hdr` debe encontrarse dentro de la carpeta `archivos_t4/` del repositorio.

El archivo `LaplacianaStarlet_2.py` contiene todas las funciones necesarias para la correcta ejecución del código. Se desarrolla la lectura de la imagen HDR, la extracción y compresión gamma del canal de luminancia, la descomposición y reconstrucción mediante pirámide Laplaciana decimada y transformada starlet no decimada (*à trous*), la aplicación de umbral duro relativo, la modificación por factores de ganancia y la visualización comparativa de resultados.


Posterior a la definición de las funciones se encuentran las líneas del código principal a ser ejecutadas.

```python
(...)
desv_est = 2.0
alpha = 0.5
factores_finos = [2.0, 2.0, 1.5, 0.8, 0.5]
factores_gruesos = [0.7, 0.8, 1.2, 1.7, 2.3]

#==========================================================Parte 0 (implementacion necesaria)==========================================================

imagen = cv2.imread("archivos_t4/belgium.hdr", cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
imagen_l = 0.177 * imagen_rgb[..., 0] + 0.813 * imagen_rgb[..., 1] + 0.011 * imagen_rgb[..., 2]
imagen_adaptada_32 = a_pantalla(imagen_l, gamma=3.2)

piramide_lap, residuo_l = piramide_laplaciana(imagen_adaptada_32, desv_est)
piramide_star, residuo_s = starlet(imagen_adaptada_32)

imagen_reconstruida_l = reconstruccion(piramide_lap, residuo_l, "Laplaciana")
imagen_reconstruida_s = reconstruccion(piramide_star, residuo_s, 'Starlet')

vlims_por_capa = [
    max(float(np.percentile(np.abs(capa_l), 99.5)),
        float(np.percentile(np.abs(capa_s), 99.5)))
    for capa_l, capa_s in zip(piramide_lap, piramide_star)
]
(...)
```

En este fragmento de código se definen los parámetros principales del procesamiento y se prepara la base común (`Parte 0`) requerida por todos los experimentos:



Consiguiente a la `Parte 0`, el código se divide en tres bloques experimentales (`Parte 1`, `Parte 2` y `Parte 3`) encargados de generar las figuras y métricas utilizadas en el informe.

```python
(...)
#==========================================================Parte 1==========================================================
# Capas de detalle, residual y verificación de reconstrucción sin modificar coeficientes
mostrar_piramide(piramide_lap, residuo_l, 'Laplaciana', desv_est, vlims_por_capa)
mostrar_piramide(piramide_star, residuo_s, 'Starlet', desv_est, vlims_por_capa)
comparar_reconstrucciones(
    [imagen_adaptada_32, imagen_reconstruida_l, imagen_reconstruida_s],
    ["Original (γ=3.2)", "Reconstruida Laplaciana", "Reconstruida Starlet"],
    "Verificación de reconstrucción sin modificar coeficientes",
    filas=1, cols=3
)

#==========================================================Parte 2==========================================================
# Umbral duro: capas umbralizadas y reconstrucción comparativa para el valor actual de alpha
mostrar_piramide(piramide_lap_umb, residuo_l, 'Laplaciana', desv_est, vlims_por_capa)
mostrar_piramide(piramide_star_umb, residuo_s, 'Starlet', desv_est, vlims_por_capa)
comparar_reconstrucciones(
    [imagen_adaptada_32, imagen_reconstruida_l_umb, imagen_reconstruida_s_umb],
    ["Original", f"Laplaciana (α={alpha})", f"Starlet (α={alpha})"],
    f"Reconstrucción tras umbral duro (α={alpha})",
    filas=1, cols=3
)

#==========================================================Parte 3==========================================================
# Modificación por ganancias (escalas finas vs. gruesas) y reconstrucción 2×2
mostrar_piramide(piramide_fina_l, residuo_l, 'Laplaciana', desv_est, vlims_por_capa)
mostrar_piramide(piramide_fina_s, residuo_s, 'Starlet', desv_est, vlims_por_capa)
mostrar_piramide(piramide_gruesa_l, residuo_l, 'Laplaciana', desv_est, vlims_por_capa)
mostrar_piramide(piramide_gruesa_s, residuo_s, 'Starlet', desv_est, vlims_por_capa)
comparar_reconstrucciones(
    [reconstruida_l_fina, reconstruida_s_fina, reconstruida_l_gruesa, reconstruida_s_gruesa],
    [
        f"Laplaciana — Finas {factores_finos}",
        f"Starlet — Finas {factores_finos}",
        f"Laplaciana — Gruesas {factores_gruesos}",
        f"Starlet — Gruesas {factores_gruesos}",
    ],
    "Modificación por ganancias (escalas finas vs. gruesas)",
    filas=2, cols=2
)

plt.show()
```

Para el uso correcto **se debe dejar descomentado únicamente el bloque o las figuras que se deseen visualizar (manteniendo siempre activa la `Parte 0`), mientras las demás figuras siguen comentadas**, y posteriormente ejecutar el código.

> Al hacer uso de un solo `plt.show()`, todas las figuras que queden activas se abrirán simultáneamente (una ventana por figura). Comentar las que no se necesiten evita sobrecargar la visualización y facilita inspeccionar los recortes ampliados con el zoom interactivo de Matplotlib.

> Las imágenes reconstruidas se grafican con escala fija (`vmin=0.0`, `vmax=1.0`) para preservar la comparabilidad de contraste y permitir inspeccionar los valores reales fuera de rango al aplicar ganancias.

En cada parte del script también se imprimen en consola las métricas cuantitativas correspondientes: el RMSE de reconstrucción exacta (`Parte 1`), el porcentaje de coeficientes conservados por capa y el RMSE tras el umbral duro (`Parte 2`), y el RMSE respecto a la original tras aplicar las ganancias finas y gruesas (`Parte 3`).

```python
print(f"α={alpha} | % conservado Lap: {np.round(porcentaje_coef_l, 2)} | Starlet: {np.round(porcentaje_coef_s, 2)}")
print(f"      RMSE vs original: Lap {rmse(imagen_adaptada_32, imagen_reconstruida_l_umb):.4f} | Starlet {rmse(imagen_adaptada_32, imagen_reconstruida_s_umb):.4f}")
```

> Estos resultados se muestran en la consola. Si se están visualizando figuras, se deben cerrar para finalizar la ejecución.

### Advertencias y consideraciones

**Para evitar abrir demasiadas ventanas a la vez, se deben comentar las llamadas a `mostrar_piramide` y `comparar_reconstrucciones` que no se deseen visualizar en esa ejecución, manteniendo siempre descomentada la `Parte 0`.**

**Para reproducir los distintos casos del experimento de umbral duro (`Parte 2`), se debe modificar la variable `alpha` al inicio del bloque principal y volver a ejecutar el código.**

**El umbral duro y los factores de ganancia se aplican exclusivamente sobre las 5 capas de detalle; el residual grueso $c_5$ se mantiene intacto en todos los experimentos.**


## Pregunta 2