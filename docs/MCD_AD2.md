# Resultados RC–MIC en la convención MCD de AD2

La pantalla de diagnóstico, el árbol de resultados y las exportaciones PDF y
CSV expresan los vectores condilares RC → MIC en esta convención:

| Componente | Positivo | Negativo |
| --- | --- | --- |
| X, sagital | Anterior (mesial) | Posterior (distal) |
| Y, transversal | Derecha del paciente | Izquierda del paciente |
| Z, vertical | Inferior | Superior |

La lectura transversal se expresa con signo positivo a la derecha del paciente;
el MCD identifica sus lecturas mediante RIGHT SHIFT / LEFT SHIFT. No se debe
invertir nuevamente el sentido por observar el movimiento de la rama superior
del instrumento: sus etiquetas ya indican el movimiento mandibular.

Referencia: manual de registro de posición condilar de AD2, revisión 03/07/11,
apartados de distracción sagital/vertical y transversal:
https://ad2mexico.com/wp-content/uploads/2025/03/AD2-MCD-Manual-Spanish-3-7-11.pdf

## Compatibilidad y alcance

La geometría y los vectores internos conservan la convención anatómica de OGDD
(+X derecha, +Y anterior, +Z superior). Se convierten exactamente una vez al
presentarlos: `(X_mcd, Y_mcd, Z_mcd) = (Y_ogdd, X_ogdd, -Z_ogdd)`.
Los archivos .ogdd existentes siguen siendo compatibles y el montaje no se
modifica. No se deben convertir manualmente los vectores almacenados.

La distancia total, los RMSE, las superficies y las transformaciones RC–MIC
permanecen iguales. Las coordenadas de selección de puntos y la orientación de
la escena 3D siguen siendo anatómicas; la nueva convención corresponde a las
lecturas clínicas del diagnóstico.

Los CSV nuevos declaran `axes,convention,AD2_MCD` y sus sentidos positivos.
Quien analice series de CSV antiguos y nuevos debe leer esos metadatos antes
de combinar componentes con el mismo nombre.

Este cambio permite comparar direcciones y signos. No corrige diferencias de
magnitud entre el montaje físico y el virtual, ni valida la exactitud clínica.
Los valores Y por cóndilo describen los puntos virtuales de OGDD: no se añade
una lectura simulada de la mesa transversal central del MCD sin conocer su
posición respecto al montaje. La orientación y posición del eje de ambos
montajes deben verificarse al estudiar su concordancia.
