# Data dictionary

Source file: `LoanExport.csv` (291,451 rows x 28 columns), an extract of the
[Freddie Mac Single-Family Loan-Level Dataset](https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset).
All loans are 30-year (mostly 360-month) fixed-rate mortgages;
99% have a first payment between February and June 1999. The repository ships a 5,000-row
stratified sample in `data/sample/`; `scripts/download_data.py` fetches the full file.

"Use" says how the column is treated by the models: **input** (origination-time feature),
**target**, **excluded** (and why).

| Column | Type | Meaning | Missing code | Use |
|---|---|---|---|---|
| `CreditScore` | int | Borrower FICO score at origination (300-850) | `0` (1,189 loans) | input |
| `FirstPaymentDate` | YYYYMM | Due date of the first scheduled payment | - | excluded, replaced by `VintageMonth` |
| `FirstTimeHomebuyer` | Y/N | First-time home buyer flag | `X` (26.8%) | input (one-hot, missing kept as a level) |
| `MaturityDate` | YYYYMM | Scheduled maturity | - | excluded (= first payment + term) |
| `MSA` | code | Metropolitan statistical area | `X` (13.4%) | input (target-encoded) |
| `MIP` | int (%) | Mortgage insurance percentage | - | input |
| `Units` | int | Number of units in the property (1-4) | `0` (2 loans) | input |
| `Occupancy` | O/S/I | Owner-occupied, second home, investment | - | input (one-hot) |
| `OCLTV` | int (%) | Original combined loan-to-value (all liens) | `0` | input |
| `DTI` | int (%) | Debt-to-income ratio at origination | `0` (6.1%) | input |
| `OrigUPB` | int ($) | Original unpaid principal balance | - | input |
| `LTV` | int (%) | Original loan-to-value (first lien) | `0` | input |
| `OrigInterestRate` | float (%) | Note rate at origination | - | input |
| `Channel` | R/B/C/T | Retail, broker, correspondent, third party (TPO not specified) | - | input (one-hot) |
| `PPM` | Y/N | Prepayment-penalty mortgage flag | `X` (1.9%) | input (one-hot) |
| `ProductType` | str | Product type | - | excluded (constant: `FRM`) |
| `PropertyState` | str | US state or territory (53 values) | - | input (target-encoded) |
| `PropertyType` | code | SF single family, PU planned unit dev., CO condo, MH manufactured, CP co-op, LH leasehold | `X` | input (one-hot) |
| `PostalCode` | code | 3-digit ZIP followed by `00` | `X` | excluded (identifier-like; MSA and state carry geography) |
| `LoanSeqNum` | str | Unique loan identifier | - | excluded (ID) |
| `LoanPurpose` | P/C/N | Purchase, cash-out refinance, no-cash-out refinance | - | input (one-hot) |
| `OrigLoanTerm` | int | Original term in months (301-361) | - | input |
| `NumBorrowers` | 1/2 | Number of borrowers | `X` | input (one-hot) |
| `SellerName` | code | Seller (originator), 2-letter abbreviation, rare sellers grouped as `Ot` | `X` (8.6%) | input (target-encoded) |
| `ServicerName` | str | Current servicer | - | excluded (servicing can be transferred after origination) |
| `EverDelinquent` | 0/1 | Loan was 30+ days delinquent at least once | - | **target 1** |
| `MonthsDelinquent` | int | Number of months reported delinquent | - | excluded: **leaks the target** (`> 0` exactly when `EverDelinquent = 1`) |
| `MonthsInRepayment` | int | Months the loan was observed in the pool (1-212) | - | excluded: post-origination; used to **define target 2** |

## Engineered columns (`mbs_risk.features`)

| Column | Definition | Use |
|---|---|---|
| `VintageMonth` | months between January 1999 and `FirstPaymentDate` | input |
| `MonthlyPayment` | level annuity payment `P r / (1 - (1 + r)^-n)`, `r = rate / 1200`, `n = OrigLoanTerm` | input |
| `TotalInterest` | `MonthlyPayment * n - OrigUPB` (interest if held to maturity) | input |
| `EstMonthlyIncome` | `MonthlyPayment / (DTI / 100)`, an upper-bound income estimate (the mortgage is only part of total debt) | input |
| `RateSpread` | `OrigInterestRate` minus the median rate of loans with the same `VintageMonth` (refinance-incentive proxy) | input |
| `CreditRange` | credit score bins `<650, 650-699, 700-749, 750+` (from the original notebook) | EDA |
| `LTV_Range` | `<=60, 61-80, 81-90, >90` | EDA |
| `DTI_Range` | `<=20, 21-30, 31-40, >40` | EDA |
| `Repay_Range` | months in repayment in 4-year bins (from the original notebook) | EDA |
| `RateBucket` | origination-rate quintiles (ties merged) | EDA |
| `Prepaid36` | `1` if `MonthsInRepayment <= 36` and `EverDelinquent = 0`; `0` if `> 36` and never delinquent; undefined for ever-delinquent loans | **target 2** |

The original notebook's LTV bins (`0-25, 25-50, 50+`) put 93% of loans in the top bin, so the
industry-standard 60/80/90 cut-offs (90+ normally requires mortgage insurance) are used instead.
