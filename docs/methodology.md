# Methodology

## 1. Questions

1. **Delinquency**: given only what is known when a loan is originated, how likely is it to
   become 30+ days delinquent at some point (`EverDelinquent`)?
2. **Prepayment**: how likely is a performing loan to be paid off early, within 36 months of its
   first payment (`Prepaid36`)? For a mortgage-backed security investor this is the core
   prepayment risk: early payoffs return principal when rates fall, exactly when reinvesting it
   is least attractive.
3. **Hypotheses**: do the six hypotheses written at the start of the original notebook (H1-H6)
   hold in the data?

## 2. Defining the prepayment target

The export has no "loan status" or "current balance" column, so prepayment has to be derived.

*What the course project suggested.* The original project brief derived a prepayment figure
from the loan terms: compute the scheduled monthly payment (EMI) from `OrigUPB`,
`OrigInterestRate` and `OrigLoanTerm`, the total payment and interest, an income estimate from
`DTI`, and a rule-based "prepayment" amount from income and DTI. Every one of these quantities is
a deterministic function of the inputs, so a model trained to predict it would only rediscover
the formula (a near-perfect score that says nothing about borrowers). The payment, interest and
income quantities are kept as **features** (`MonthlyPayment`, `TotalInterest`, `EstMonthlyIncome`),
not as a target.

*Definition used here.* `MonthsInRepayment` is the number of months a loan was observed in the
pool. The longest observed life is 212 months for a loan with a 1999-03 first payment, i.e. the
data runs to about November 2016, so every loan had at least 36 months in which it could be
observed. A loan that disappears earlier has therefore been terminated, not censored. A
termination is a voluntary **prepayment** (refinance or home sale) unless the loan defaulted;
the export does not say which, but a loan that was never delinquent cannot have defaulted. Hence:

```
Prepaid36 = 1   if EverDelinquent == 0 and MonthsInRepayment <= 36
Prepaid36 = 0   if EverDelinquent == 0 and MonthsInRepayment >  36
undefined       if EverDelinquent == 1          (early exit might be a default)
```

Assumptions, stated plainly:

* the end of the observation window is inferred from the data (max loan life), not documented
  in the file;
* excluding ever-delinquent loans makes the prepayment model a model of *performing* loans; it is
  not applicable to loans that are already delinquent;
* 36 months is a modelling choice (it splits performing loans 30.5% / 69.5% and matches the
  period in which most 1999 loans refinanced during the 2001-2003 rate drop); `add_targets`
  takes the horizon as a parameter.

## 3. Leakage control

| Column | Why it is excluded from model inputs |
|---|---|
| `MonthsDelinquent` | `MonthsDelinquent > 0` is identical to `EverDelinquent == 1` (checked: 0 disagreements in 291,451 loans). The original notebook kept it in the correlation and pair plots. |
| `MonthsInRepayment` | Observed after origination; it defines `Prepaid36` and is also mechanically related to delinquency (longer-lived loans have more months in which to miss a payment). |
| `ServicerName` | The servicer can change after origination (servicing transfers). |
| `LoanSeqNum`, `PostalCode` | Identifier-like; geography is represented by state and MSA. |
| `ProductType`, `MaturityDate`, `FirstPaymentDate` | Constant / redundant (replaced by `VintageMonth`). |

`mbs_risk.features.model_frame` refuses to build an input matrix that contains any of these
columns, and a test checks it.

**Encoding leakage in the original notebook.** The M-estimate target encoder was fitted on a
random 20% "encoding split", and the remaining 80% was later split again into train and
validation; target statistics were also computed before the train/validation split of the
final pipeline. Here every transformer lives inside one scikit-learn `Pipeline` fitted on the
training rows only, and the target encoder (`sklearn.preprocessing.TargetEncoder`) cross-fits
its statistics (5 folds) so that each training row is encoded without its own label.

## 4. Cleaning

* `"X"` (with or without padding) is the "not available" code in string columns; zeros are
  impossible for credit score, LTV, OCLTV, DTI and units. Both become `NaN`.
* No duplicate rows exist; the check stays in the pipeline.
* Outliers are **not** removed for modelling. The original IQR filter (whose lower fence was
  the builtin `min` function by mistake) is fixed in `cleaning.remove_outliers_iqr` and used for
  EDA only: removing rows from the test set would make the evaluation optimistic, and tree
  models are insensitive to extreme values.
* Imputation (inside the pipeline): median + missing-indicator for numeric columns, constant
  `"missing"` level for categorical columns.

## 5. Models and evaluation

* Split: stratified 80/20 train/test; the training part is split again 80/20 into fit and
  validation (64/16/20 overall). Seed 42.
* Models: majority-class (prior) baseline, logistic regression, random forest (200 trees,
  `min_samples_leaf=20`, `max_samples=0.5`), histogram gradient boosting (early stopping on an
  internal 10% split), XGBoost (600 trees, depth 6, learning rate 0.05, subsampling).
* Metrics on the untouched test set: ROC-AUC (ranking), PR-AUC / average precision (more
  informative with 20-30% positives), Brier score (calibration), and precision / recall / F1
  at the threshold that maximises F1 on the validation split (the default 0.5 is arbitrary for
  imbalanced classes). 3-fold stratified cross-validated ROC-AUC on train+validation is reported
  for logistic regression, gradient boosting and XGBoost to show the variance of the estimate.
* Calibration: reliability curves on quantile bins.
* Explanation: permutation importance of the best model on 20,000 test loans (drop in ROC-AUC
  when an input column is shuffled; 3 repeats). It is computed on raw input columns, so the
  one-hot and target-encoded columns are explained as a whole.

## 6. Hypothesis tests

The hypotheses concern prepayment, so they are tested with `Prepaid36` on never-delinquent loans
(233,788 loans). With this sample size every difference is "significant" (chi-square p-values
below 1e-5), so the verdict is driven by effect size: **supported** if the prepayment rate moves
in the claimed direction by at least 2 percentage points between the extreme groups,
**contradicted** if it moves the opposite way by at least 2 points, otherwise **no material
effect**. These are univariate associations, not causal effects; for example, low credit
scores come with higher note rates, so H1 and H6 are partly the same signal.

## 7. Limitations

* One origination cohort (1999) dominates the data; results describe how 1999 loans behaved
  through the 2001-2003 refinancing wave and the 2008 crisis, not a general law.
* No macroeconomic or time-varying data (market rates, house prices, unemployment). Prepayment is
  mostly driven by the market rate *after* origination, which a static origination model cannot
  see; this caps achievable accuracy.
* `EverDelinquent` mixes "how risky is the borrower" with "how long was the loan observed": a
  loan that prepaid after 20 months had less time to become delinquent.
* The prepayment target cannot separate refinancing from home sales or curtailments.
