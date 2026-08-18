\# TB Kiribati Julia



A lightweight Julia implementation of an age-structured tuberculosis

transmission model for Kiribati.



The project is being developed as an independent implementation of a model

originally constructed using the R package `agepi`.



\## Current status



Phase 1 implements:



\- 96 one-year age groups

\- 10 epidemiological compartments

\- age-specific susceptibility and progression

\- age-structured contact mixing

\- tuberculosis natural-history transitions

\- treatment

\- disease-induced mortality

\- age-resolved cumulative epidemiological flows

\- deterministic simulation using the SciML ecosystem



Demographic coupling using UN World Population Prospects 2024 data will be

added in subsequent development.



\## Running the model



From the project directory:



```julia

julia --project=. scripts/run\_model.jl

```



\## Running the model

```julia

julia --project=. test/runtests.jl

```





