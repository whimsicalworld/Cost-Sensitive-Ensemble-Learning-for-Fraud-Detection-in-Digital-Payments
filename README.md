<div align="center">Fraud Detection in Digital Payments
Cost-sensitive ensemble learning on the IEEE-CIS Fraud Detection benchmark




Background · Dataset · Methodology · Results · Governance · Run the notebook · Future work
</div>Overview
This MSc Data Science (FinTech) project investigates how to flag fraudulent e-commerce payments while limiting false alerts on legitimate transactions. It builds an offline fraud-scoring pipeline and compares a Decision Tree baseline with four cost-sensitive tree ensembles.

The final Random Forest achieved 0.6377 held-out PR-AUC, up from 0.3523 for the Decision Tree baseline. At the evaluated cutoff, about 85% of flagged transactions were fraudulent, and the model detected 45% of actual fraud cases.
Background and literature
The digital payments problem
An e-commerce payment typically takes place without the merchant physically inspecting a card. The same transaction record can contain a payment amount, product information, email domain, and clues about the device or browser used at checkout. Individual fields rarely tell the whole story. A device can be unfamiliar because a legitimate customer has bought a new phone. A high-value purchase can be routine for one card and unusual for another. Context across fields and transactions is therefore central to the project’s definition of fraud risk.

The dissertation describes the evolution from straightforward misuse of stolen credentials to coordinated activity across devices, accounts and channels. It also discusses how customer behaviour changes over time. An effective research design must account for both phenomena. Relationships visible in a historical dataset may weaken as payment products, merchant populations or fraud patterns change. The IEEE-CIS benchmark provides an empirical setting for studying classification under these conditions. Its anonymised fields support pattern discovery, with limited grounds for interpreting why a specific pattern exists or transferring performance conclusions to UK payment systems.
From rules to ensemble models
Rules offer a transparent way to act on known conditions, such as a combination of transaction value and recent activity. Analysts can inspect the trigger and update it when business policy changes. Maintaining broad coverage becomes difficult as the number of interacting conditions grows, and new behaviours call for regular reassessment. Bhattacharyya et al. (2011) provide comparative evidence for machine-learning approaches to credit card fraud. Breiman (2001) provides the basis for the Random Forest approach explored here. The dissertation also draws on scalable gradient boosting research to motivate XGBoost, LightGBM and CatBoost.

The research question is whether a weighted ensemble can improve the ranking of fraudulent transactions over a single decision tree under the project’s data preparation and validation design. Cost sensitivity in this experiment comes from class weights. A separate operational question is how to translate those scores into actions such as additional verification or analyst review. That choice depends on the costs and consequences of errors, the capacity of an investigations team, and the quality of the estimated probabilities.
The role of explainability
The dissertation discusses SHAP and LIME as ways to inspect the factors behind model predictions. A global summary can show the features the model relies on across many observations. A local explanation can help a reviewer understand the score assigned to a particular transaction. Explanations require their own validation, especially where multiple correlated features encode similar information. The implemented notebook produces model scores and classifications. Explanation generation, explanation stability tests and analyst-facing reason codes remain research extensions.
Dataset
The IEEE-CIS Fraud Detection benchmark, provided by Vesta Corporation through Kaggle, contains card-not-present e-commerce transactions. The project joins transaction and identity records on TransactionID and predicts the binary target isFraud.
Dataset profile	Value
Labelled transactions	590,540
Fraudulent transactions	20,663
Legitimate transactions	569,877
Fraud prevalence	3.50%
Predictors after feature engineering and preprocessing	456


Composition and exploratory analysis
The initial merge produced 434 columns. Parsing device and identity fields brought the count to 442, after which 13 sparse columns were removed, leaving 429 before further feature construction. The dissertation reports zero duplicate rows in the initial checks. The final 456 predictors comprise 416 numerical and 40 categorical features prior to their modelling representation. Input families include payment amount and relative time, product and card attributes, addresses and distances, email domains, identity and device fields, and anonymised count, timing, match and V-series variables.

The exploratory analysis reviewed missingness, data types, summary statistics, correlations and class frequencies. Numerical outliers were flagged using the 1.5 × interquartile range rule. Among the highest reported outlier shares were V221 at 9.94%, card1_count_full at 9.78%, V295 and V318 at 9.39% each, and V133 at 9.19%. Extreme observations were retained because unusual behaviour can be informative for fraud modelling. The presence of an outlier by itself says little about whether a payment was fraudulent.
Labelled split	Legitimate	Fraudulent	Total
Development	455,902	16,530	472,432
Held-out	113,975	4,133	118,108
All labelled observations	569,877	20,663	590,540


The notebook also loads the competition test data. The metrics in this README come from a held-out portion of the labelled training data. Keeping those two uses of “test” distinct matters when checking data paths and interpreting the results.
Methodology
Stage	Implementation
Data preparation	Join transaction and identity records, clean device fields, impute missing values, and remove variables with more than 90% missingness.
Feature engineering	Build transaction amount and time features, extract device and browser attributes, and derive interaction, frequency and group-statistic features.
Memory optimisation	Downcast data types to reduce the reported memory footprint from 2,054 MB to 514 MB, approximately 75%.
Model comparison	Use a Decision Tree baseline and compare Random Forest, XGBoost, LightGBM and CatBoost with random stratified five-fold cross-validation.
Hyperparameter search	Evaluate six Random Forest configurations with RandomizedSearchCV across five folds.
Held-out evaluation	Use a random stratified 80:20 split. Report PR-AUC as the primary metric and use fraud precision, recall and F1 to assess the evaluated cutoff.


The final Random Forest uses 500 trees, max_depth=30, max_features="log2", min_samples_leaf=1 and class_weight="balanced_subsample".
Architecture and implementation
The pipeline begins with CSV files for training transactions, training identities and their competition test counterparts. The notebook also reads submission data. A left join preserves every payment from the transaction table even when the matching identity record is absent. The resulting tables are assessed for data quality and schema consistency before feature engineering. After separating isFraud and removing transaction identifiers from the predictors, the notebook trains models on the prepared feature matrix. It then generates fraud scores, applies a classification threshold and appends predictions to the competition test output.
Input or field	Derived representation
DeviceInfo	Device name and version with common manufacturer variants grouped
id_30, id_31, id_33	Operating system and version, browser and version, and screen width and height
id_23 and id_34	Identity values extracted from prefixed strings
TransactionAmt	Logarithmic and decimal components, plus group-relative amount features
TransactionDT	Estimated hour of day and day of week derived from relative time
id_02 and D15	Ratios based on group means and spreads
Card, address, device and email fields	Cross-field combinations, category counts and broader email domain groups


The clean_device_info function uses a safe_split helper to handle uneven strings and absent fields. Group statistics use contexts such as card and address categories to express whether a value is high or low relative to comparable observations. Count features give the model an estimate of how common a category is, and pairwise interactions capture combinations the model might otherwise have to reconstruct through many splits. These features describe patterns in the available data. Their prospective usefulness depends on whether each transformation could be computed using information available when a payment arrives.
Missingness, encoding and computational constraints
Sparse identity coverage is expected after a left join. The project checks missingness in both datasets and drops the union of fields that exceed 90% missingness. Remaining numerical gaps are filled with median values, with infinite values converted to missing first. Categorical gaps are represented by an explicit missing category before label encoding. The implementation description covers robust scaling for numerical inputs, while the report’s preprocessing discussion also records standard and min-max scaling stages in the experimental workflow. Tree splits generally need little benefit from repeated monotonic scaling, so this is an area for simplification and precise notebook-level documentation.

Memory becomes material when nearly 600,000 observations are kept through hundreds of feature transformations and repeated model fits. The reduce_mem_usage routine uses smaller compatible types to lower the size of the feature matrix. Parallel execution in Random Forest helps distribute the work across available CPU cores. Downcasting addresses memory consumption, while parallel execution addresses compute time. Together they make the notebook experiments easier to run on a limited machine.
How the models learn under class imbalance
A Decision Tree provides an interpretable baseline for the effects of recursive partitioning. A deep tree can create small leaves around individual fitting examples, which is consistent with the baseline’s perfect development score and sharp decline on held-out data. Random Forest reduces dependence on any one tree by training many trees on bootstrap samples and considering a random subset of predictors at each split. It aggregates the trees’ estimated class probabilities to produce a fraud score. This can reduce variance and capture different combinations of transaction, identity and device signals.

Random Forest uses balanced_subsample to weight classes within each bootstrap sample. XGBoost and LightGBM grow trees sequentially, using the current loss to guide subsequent updates, and use scale_pos_weight to increase the influence of fraud examples. CatBoost also uses boosting and applies auto_class_weights="Balanced" in this experiment. The legitimate-to-fraud ratio is about 27.58 to 1. The pipeline retains the original observations throughout this comparison. The report discusses random undersampling and SMOTE as alternatives, noting that undersampling can discard useful legitimate behaviour and interpolation can create questionable combinations of encoded or masked features. A controlled comparison of resampling strategies remains open.
Metric and threshold design
Precision is the fraction of raised alerts that correspond to labelled fraud. Recall is the fraction of labelled fraud the model catches. F1 combines both at a specific cutoff. The report uses precision–recall area (PR-AUC) as its main ranking measure and discusses ROC-AUC as supplementary context. Precision–recall evaluation makes it easier to see how effectively the model surfaces positives in a dataset where most payments are legitimate. An always-legitimate classifier could reach roughly 96.5% accuracy while catching zero fraud.

During experimentation, some threshold exploration used F1 on training predictions. The reported precision and recall describe an evaluated operating point. They do not identify a cutoff justified by expected losses, customer impacts or service capacity. A prospective workflow would reserve distinct data for fitting, probability calibration, threshold selection and final evaluation. For a payment-review queue, it could also report recall at a fixed alert volume so the scorecard reflects the number of cases investigators can actually handle.
Results
Held-out performance
Metric	Decision Tree	Final Random Forest
PR-AUC	0.3523	0.6377
Fraud precision	0.5534	0.85
Fraud recall	0.6121	0.45
Fraud F1-score	0.5813	~0.59
Accuracy (secondary)	96.91%	97.80%


The Random Forest produced a higher proportion of correct alerts, while the Decision Tree caught a larger share of fraud cases. At the reported Random Forest cutoff, approximately 55% of fraudulent transactions went undetected. With fraud representing only 3.50% of the dataset, accuracy alone gives little insight into this trade-off.
<details>
<summary><strong>Five-fold ensemble comparison</strong></summary>Model	Mean CV PR-AUC
Random Forest	0.7347
XGBoost	0.6994
LightGBM	0.6218
CatBoost	0.6082


The comparison reflects the shared preprocessing. CatBoost received label-encoded inputs, so its native categorical processing was unused.
</details>Testing, interpretation and implementation challenges
The five-fold results measure ranking on random stratified folds. Random Forest had a reported standard deviation of 0.0052 and a best single-fold score of 0.7439. These results supported its selection for final evaluation. The final holdout uses a separate comparison in the report. Relative to the baseline, the Random Forest improved held-out PR-AUC by 0.2854 and fraud precision by roughly 30 percentage points. Recall decreased by roughly 16 percentage points. The choice of model and operating point would therefore affect both the review queue and the number of fraudulent transactions missed.

The experiment faced six intertwined challenges. Memory downcasting made the engineered data manageable. Device parsing and median or category-based imputation addressed sparse, inconsistent identity records. Aligning the feature schemas allowed the prepared datasets to pass through the same model inputs. Class weighting and PR-AUC helped keep the rare target visible during evaluation. Ensemble learning improved held-out ranking over the high-variance baseline, with some overfitting remaining. Finally, a six-candidate search kept the tuning budget tractable at 30 fits, while the difference between searched and final tree depth showed why the evaluation should preserve each configuration alongside its score.

The retained features and model choice should be read in the context of the experimental design. Some exploratory comparisons used predictions on fitting data. Some feature encodings and aggregates incorporated competition test features. The report attributes the development-to-holdout gap partly to residual overfitting. A future replication could document which features and preprocessing stages contribute most to the held-out result through controlled ablations performed with an out-of-time split.
Run the notebook
1. Download the competition CSVs from the Kaggle data page. The labelled training files are train_transaction.csv and train_identity.csv. Keep the competition test CSVs available for notebook cells that reference them.
2. From a Python environment, install the core notebook dependencies and launch JupyterLab.
   python -m pip install jupyterlab numpy pandas scikit-learn matplotlib seaborn xgboost lightgbm catboost
   jupyter lab
3. Open the project’s fraud-detection notebook, update its CSV paths to your local files, and run the cells in order. Compare the final metrics with the held-out results above.
Limitations
- Validation: The random folds and holdout measure performance within the same dataset. A chronological split would evaluate how the model handles later fraud patterns.
- Preprocessing: Some encoders, count mappings and group statistics use combined feature distributions. This can make reported performance optimistic relative to a pipeline that fits transformations on training data only.
- Threshold selection: Some cutoff choices use training predictions. A separate validation set and explicit costs for missed fraud and false alerts would support an operational cutoff.
- Reproducibility: The search selected max_depth=None. The final experiment used max_depth=30. Development PR-AUC was 0.9813, compared with 0.6377 on the holdout, indicating residual overfitting.
Legal, social, ethical and professional issues
The dissertation considers how a fraud scoring model could affect customers if it were integrated into a UK payment environment. This section summarises the report’s governance discussion. The project itself uses de-identified benchmark data in an offline experiment.
Topic	Issue considered in the dissertation
Privacy and lawful processing	The report discusses UK GDPR, the Data Protection Act 2018 and the Data (Use and Access) Act 2025 in relation to lawful purpose, data minimisation, retention, access and security. It proposes assessing data protection impacts before applying similar profiling to customer data.
Automated decisions and recourse	A risk score could inform additional authentication or an analyst review. The report considers how people would receive a clear explanation, challenge an adverse outcome and reach a human reviewer.
Equality and discrimination	Device, postcode or email attributes can serve as proxies. The report considers the Equality Act 2010 and the need to investigate differences in errors and customer outcomes. The benchmark lacks reliable protected-group labels for formal demographic fairness evaluation.
Financial regulation and consumer protection	The report considers the FCA Consumer Duty when discussing foreseeable harm, fair communications and the customer impact of blocked or delayed payments. It proposes reviewing the action taken at each score level as part of threshold governance.
Professional accountability	The report relates the BCS Code of Conduct to privacy, competence and transparent representation of the system. Developers, deployment approvers and payment investigators would each have defined responsibilities.


Social impact and ethical interpretation
Errors can have different consequences depending on the transaction and the person affected. A false positive might delay payment for travel, food or medicine. A false negative might allow a fraudulent charge to proceed and increase recovery time. Aggregate recall and precision do not show whether the same people repeatedly experience disruption. The dissertation proposes examining outcomes for vulnerable customers and giving customers a plain-language account of security checks and routes to resolution. Analysts would need more detailed evidence to review a case.

There are limits to what the data can tell us about a person. Historic labels are shaped by the cases that were detected, investigated and recorded. Missing identity attributes may be unevenly distributed, and imputing a value can conceal that uncertainty in the final feature matrix. Privacy and fairness analysis would need to examine whether specific inputs are necessary and how their errors are distributed. Local model explanations would require stability checks and expert review before being used to support decisions about customers.
Security and operational accountability
The report identifies secure access, versioned preprocessing, model documentation, decision logging, analyst review and a rollback process as components of a governed operational system. It also identifies security risks such as unauthorised access and manipulation of model inputs. The notebook does not evaluate those operational controls. A deployment assessment would need to record who approved each model and threshold, how review decisions can be audited, and how a change in input or outcome distributions would be investigated.
Future work
Research direction	Proposed next experiment
Advanced ensemble architectures	Compare stacking and blending with the single Random Forest. Build meta-model inputs from out-of-fold predictions and compare on the same temporal holdout.
Cost-based decision thresholds	Calibrate scores and estimate fraud losses, false alerts and investigation capacity at alternative cutoffs. Choose the cutoff on separate validation data.
Temporal validation	Train on earlier periods and evaluate on later periods. Fit encodings, counts and group aggregates on historical observations with explicit fallbacks for unseen categories.
Graph-based detection	Explore connections among transactions, merchants, cards and devices. Check whether graph features capture patterns missed by individual transaction records.
Sequential behavioural modelling	Test whether ordered activity and changes in time, amount or merchant mix improve detection. The dissertation proposes LSTMs, temporal convolutional networks and Transformers as candidate methods.
Explainable AI	Implement SHAP for global feature summaries and individual alert explanations. Review the stability and plausibility of explanations before deriving analyst-facing reason codes.
Online learning and drift detection	Monitor shifts in inputs, score distributions and labelled outcomes over time. Evaluate retraining triggers using delayed fraud labels and a governed release process.
Privacy-preserving collaboration	Explore federated learning, differential privacy and secure multiparty computation for cross-institution studies. Measure the trade-offs among data exposure, performance and implementation cost.
Real-time production systems	Prototype a feature store and scoring API, then measure inference latency, throughput, availability and investigation workload before live integration.


Conclusion and professional development
The dissertation demonstrates a complete offline experiment, from integrating large transaction and identity files to evaluating a cost-sensitive fraud classifier. Its strongest held-out result is the Random Forest’s improvement in PR-AUC over the baseline. The low recall at the evaluated threshold limits the application of that result to an alerting system. The priority for a follow-up study is to establish whether the ranking improvement persists on genuinely later data with historically fitted features and a separately selected cutoff.

The project developed practical skills in Python data engineering, tabular feature design, imbalanced classification, ensemble learning, hyperparameter search and memory optimisation. It also strengthened the connection between technical choices and their consequences. A statistically better ranking can still correspond to a weak decision policy, especially where missing a fraudulent charge and delaying a legitimate purchase create different kinds of harm. Recording the validation design, preprocessing scope and exact final hyperparameters is part of making that judgement reproducible.
Selected references
- Kaggle and Vesta (2019). IEEE-CIS Fraud Detection competition.
- Breiman (2001). Random Forests.
- Bhattacharyya et al. (2011). Data mining for credit card fraud: A comparative study.
- Chen and Guestrin (2016). XGBoost: A Scalable Tree Boosting System.
- Dal Pozzolo et al. (2018). Credit Card Fraud Detection: A Realistic Modeling and a Novel Learning Strategy.
- Saito and Rehmsmeier (2015). The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets.
Author
Rahul Kante · MSc Data Science (FinTech), University of Greenwich
