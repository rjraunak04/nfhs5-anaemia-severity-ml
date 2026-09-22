# Data

## Data source

This project uses the India National Family Health Survey 2019–2021
(NFHS-5) Individual Recode data obtained through the DHS Program.

The respondent-level NFHS data are not distributed with this repository.
Users must request access directly from the DHS Program and comply with
the applicable data-use agreement.

## Private data files

The following files must remain outside Git version control:

- `nfhs_final.parquet` — full source dataset
- `anaemia_women_research_40cols.csv` — selected research dataset
- Processed participant-level datasets
- Train, calibration and test participant identifiers
- Model predictions and SHAP values linked to respondents

## Expected 40-column dataset

The modelling pipeline expects these variables:

### Survey design and geography

`v002`, `v003`, `v005`, `v021`, `v022`, `v024`, `sdist`

### Socio-demographic variables

`v012`, `v025`, `v501`, `v130`, `s116`, `v190`, `v133`

### Fertility and reproductive variables

`v201`, `v208`, `v212`, `v213`, `v228`, `v312`, `v404`, `v405`

### Anthropometry, healthcare access and environment

`v445`, `v467b`, `v467c`, `v467d`, `v467f`, `v113`, `v116`,
`v161`, `v481`

### Nutrition and comorbidity variables

`s728a`, `s731b`, `s731c`, `s731d`, `s731e`, `s731f`, `s731g`

### Outcome variables

- `v456` — adjusted haemoglobin; outcome sensitivity analysis only
- `v457` — primary multiclass anaemia-severity target

Neither `v456` nor `v457` may be used as a model predictor.

## Local/Drive storage

A recommended private storage structure is:

```text
nfhs5_anaemia_private/
├── data/
│   ├── raw/
│   ├── selected/
│   └── processed/
├── runs/
└── final_outputs/