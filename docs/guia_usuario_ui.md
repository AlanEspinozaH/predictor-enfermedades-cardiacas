# Guía de uso de la interfaz

## Inicio

```powershell
python -m streamlit run src/app.py
```

## Procedimiento

1. Verifique que la aplicación confirme el artefacto del manifiesto.
2. Complete las 27 variables; `HDL` es obligatorio.
3. Respete las unidades mostradas.
4. Ejecute la clasificación.
5. Interprete el resultado únicamente como salida del prototipo heredado.

## Interpretación correcta

La probabilidad mostrada corresponde a la clase positiva aprendida por el
modelo para un antecedente autorreportado. No expresa probabilidad de sufrir un
infarto en un periodo futuro.

## Errores esperados

La aplicación se detiene si:

- falta el modelo o la configuración;
- un hash no coincide;
- el esquema del pipeline contradice el contrato;
- falta una característica;
- aparece una variable obsoleta;
- un valor está fuera del rango aceptado;
- el modelo devuelve probabilidades inválidas.
