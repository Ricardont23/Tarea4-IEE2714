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

El archivo `ColorHDR.py` contiene todas las funciones necesarias para la correcta ejecución del código. Se desarrolla la lectura de una imagen HDR, el cálculo de su luminancia, la compresión de rango dinámico mediante el filtro homomórfico y el método de Durand, y la reconstrucción del color con dos estrategias distintas (razón de cromaticidad y ajuste de saturación en HSV), además de la comparación contra el procesamiento independiente por canal RGB.

Posterior a la definición de las funciones se encuentran las líneas del código principal a ser ejecutadas.

```ruby

(...)

ruta_imagen = "archivos_t4/belgium.hdr"

params_homo = dict(gamma_H=1.5, gamma_L=0.5, D0=30.0, c=1.0)
params_durand = dict(sigma_s=16.0, sigma_r=0.4, contraste_objetivo=5.0)

betas = [0.6, 1.0, 1.4]
alphas = [0.6, 1.0, 1.4]

imagen_bgr = cv2.imread(ruta_imagen, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
imagen_rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)
imagen_orig_vis = normalizar_rango(np.maximum(imagen_rgb, 0.0), gamma=2.2)

Y = calcular_luminancia(imagen_rgb)

Y_homo = normalizar_rango(filtro_homomorfico(Y, **params_homo), gamma=2.2)
Y_dur = normalizar_rango(metodo_durand(Y, **params_durand), gamma=2.2)

rgb_homo_indep = hdr_por_canal_rgb(imagen_rgb, filtro_homomorfico, **params_homo)
rgb_dur_indep = hdr_por_canal_rgb(imagen_rgb, metodo_durand, **params_durand)

(...)
```
En este fragmento se define la imagen HDR a utilizar, los parámetros de ambos métodos de compresión (`params_homo` y `params_durand`) y los valores de β y α a evaluar. Luego se calcula la luminancia comprimida con cada método y la versión procesada independientemente por canal RGB.

> La imagen `belgium.hdr` debe encontrarse dentro de la carpeta `archivos_t4` del repositorio, o bien se debe modificar `ruta_imagen` según corresponda.

### Parte 1: Estrategia 1 (razón de cromaticidad)

```ruby
(...)

for b in betas:
    img_h, clip_h = reconstruir_color_beta(imagen_rgb, Y, Y_homo, beta=b)
    met_h = metricas_color(imagen_rgb, img_h)
    (...)
    img_d, clip_d = reconstruir_color_beta(imagen_rgb, Y, Y_dur, beta=b)
    met_d = metricas_color(imagen_rgb, img_d)
    (...)

(...)
```
Para cada valor de β se reconstruye el color con ambos métodos, se imprime en consola el porcentaje de clipping junto con las métricas de color y se genera una figura de 1x3 por método.

### Parte 2: Estrategia 2 (ajuste de saturación HSV)

```ruby
(...)

for a in alphas:
    img_h, clip_h = reconstruir_color_hsv(imagen_rgb, Y, Y_homo, alpha=a)
    met_h = metricas_color(imagen_rgb, img_h)
    (...)
    img_d, clip_d = reconstruir_color_hsv(imagen_rgb, Y, Y_dur, alpha=a)
    met_d = metricas_color(imagen_rgb, img_d)
    (...)

(...)
```
Análogo a la Parte 1, pero variando el factor de saturación α. También se imprimen las métricas por consola y se generan las figuras correspondientes.

### Parte 3: Comparaciones

```ruby
(...)

print("[Homomorfico | RGB indep]", metricas_color(imagen_rgb, rgb_homo_indep))
print("[Durand | RGB indep]", metricas_color(imagen_rgb, rgb_dur_indep))

(...)

recorte_color = (200, 450, 50, 300)
recorte_borde = (150, 400, 320, 570)

(...)
```
Se imprimen las métricas del procesamiento independiente por canal y se generan dos figuras de comparación principal (una por método) con la imagen original, la luminancia comprimida, el resultado RGB independiente y las mejores variantes de cada estrategia. Finalmente, se generan recortes ampliados de dos zonas de interés: una con colores intensos (`recorte_color`) y otra con una transición fuerte de iluminación (`recorte_borde`). Ambos se definen como `(y1, y2, x1, x2)` en píxeles.

### Salida por consola

Por cada configuración se imprime una línea con el formato:

```ruby
[Metodo | parametro] clip=XX.XX% | {S_in_media, S_out_media, C_out_media, delta_H_media_deg, delta_H_p95_deg}
```

### Advertencias y consideraciones

**Todas las figuras se muestran al final con un único `plt.show()`, por lo que se abrirán varias ventanas a la vez (2 de la Parte 1, 2 de la Parte 2, 2 comparaciones principales y 2 recortes). Se recomienda cerrarlas una a una para revisarlas.**

**Los recortes `recorte_color` y `recorte_borde` están definidos para las dimensiones de `belgium.hdr`. Si se usa otra imagen, hay que ajustarlos para que queden dentro de sus límites.**

**La función `comparar_imagenes` requiere que `filas * cols` sea al menos igual a la cantidad de imágenes entregadas; de lo contrario, las imágenes sobrantes no se mostrarán.**

