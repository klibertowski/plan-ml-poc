# Data contract and collection starting point

## Implemented synthetic table

One row represents a case. All fields are required by the POC; missing or nonfinite numeric values raise an error. This strict behavior avoids silently pretending an imputation or missing-outcome policy exists.

| Column | Role | Units / meaning |
|---|---|---|
| `patient_id` | Grouping only | Pseudonymous identifier; repeated procedures stay together |
| `age` | X | Years at decision time |
| `bmi` | X | kg/m² at decision time |
| `baseline_function` | X | Generic invented 0–100 preoperative function measure |
| `preop_hka_deg` | X | Pretreatment HKA deviation, arbitrary positive-varus convention |
| `mldfa_deg` | X | Pretreatment distal femoral angle in degrees |
| `mpta_deg` | X | Pretreatment proximal tibial angle in degrees |
| `surgeon_b` | W | Binary synthetic surgeon indicator, 0 or 1 |
| `hospital_b` | W | Binary synthetic hospital indicator, 0 or 1 |
| `femoral_varus_deg` | T | Executed angle, degrees, positive varus |
| `tibial_varus_deg` | T | Executed angle, degrees, positive varus |
| `tibial_slope_deg` | T | Executed posterior slope, degrees |
| `synthetic_fjs_12mo` | Y | Invented outcome on a 0–100 scale, higher better |

Inference requires a one-row patient table with X and W; candidate/reference vectors are supplied separately in the exact declared `PLAN_COLUMNS` order. The library selects table fields by name rather than incidental CSV order. An inference patient does not need an outcome or historical treatment.

The current executable data path is **synthetic only**. `fit_models` deliberately expects the synthetic target name. The following table is a collection design starting point, not a claim that arbitrary clinical CSVs are already supported.

## Plan a real dataset before adding a real-data loader

“Required” depends on the eventual causal question and decision time. These priorities summarize the project discussions; a clinical/statistical review must finalize the protocol.

| Priority | Data to capture | Why |
|---|---|---|
| Required | Patient, joint, case IDs; side; primary/revision status; procedure and decision dates | Define eligible cases, correct linkage and independent splits |
| Required | Pretreatment CT/WLR or validated derived anatomy, units, coordinate frames, extraction versions | Reproducible patient geometry and effect modifiers |
| Required | Proposed, surgeon-approved and actually executed plans as separate records | Define the intervention and quantify execution differences |
| Required | Component orientations/positions, implant model/system/size, alignment intent and important co-interventions | Treatment consistency; avoid relying solely on an MA/KA label |
| Required | Actual outcome instrument, score, survey date, baseline function and follow-up status | Define target/time window; avoid substituting one PROM for another |
| Required | Relevant pretreatment clinical confounders: age, BMI, baseline pain/function, comorbidity, deformity/severity; surgeon/site/era and technique availability | Address treatment-selection and outcome causes at the defined decision time |
| Required | Missingness, reason for absent follow-up, measurement provenance | Prevent zero-filling outcomes and assess selection bias |
| Required for meaningful plan validity | Physical geometry, implant fit, resection/coverage/impingement and biomechanical constraints appropriate to the intended procedure | Enforce meaningful feasibility independently of ML benefit |
| High-value extension | Pretreatment expectations, mental health, activity, pain sensitization, ROM/ligament status | Potential confounders/effect modifiers when relevant and measurable |
| High-value extension | Complications, revisions, dates and censoring | Add safety/durability endpoints using appropriate analyses |
| Later extension | Gait, wearables, longitudinal recovery, rehab adherence | Secondary outcomes/mechanisms; postoperative variables are not baseline confounders |

Do not merge planned angles and postoperative measured angles into one unlabeled field. The POC assumes requested settings are executed exactly; real use needs an explicit treatment-fidelity/execution model. Use only information available at the modeled decision time in baseline inputs. Whether an intraoperative measurement is pretreatment depends on that decision time.

A future loader should validate a versioned export, preserve a missingness report, separate preprocessing fits from evaluation, maintain patient/joint groups and reserve time/site holdouts. Do not rename real outcomes to `synthetic_fjs_12mo` and present the resulting fit as a validated study.
