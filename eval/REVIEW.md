# Eval set review

93 questions, built 2026-11-04 by `diligence-agent build-eval`; gold answers come from the sources in code. Check the sample below against the linked records and mark each row. Fix wording or drop a question in `src/diligence_agent/evalset.py`, then rebuild.

## Sample to check (20)

| OK? | ID | Split | Question | Gold | Check | Sources |
|---|---|---|---|---|---|---|
| [ ] | pat08 | dev | When does Voranigo lose its last Orange Book patent protection? | Patent 11345677 expires 2039-01-16. | date | [1](https://www.accessdata.fda.gov/scripts/cder/ob/results_product.cfm?Appl_Type=N&Appl_No=218784) |
| [ ] | pat09 | test | When does the last-expiring FDA Orange Book patent listed for Cabometyx expire? | Patent 9724342 expires 2033-07-09. | date | [1](https://www.accessdata.fda.gov/scripts/cder/ob/results_product.cfm?Appl_Type=N&Appl_No=208692) |
| [ ] | exc04 | test | What is the latest-expiring FDA regulatory exclusivity listed for Myqorzo in the Orange Book, and when does it expire? | Exclusivity NCE expires 2030-12-19. | date | [1](https://www.accessdata.fda.gov/scripts/cder/ob/results_product.cfm?Appl_Type=N&Appl_No=219083) |
| [ ] | exc06 | dev | What is the latest-expiring FDA regulatory exclusivity listed for Evrysdi in the Orange Book, and when does it expire? | Exclusivity ODE-400 expires 2029-05-27. | date | [1](https://www.accessdata.fda.gov/scripts/cder/ob/results_product.cfm?Appl_Type=N&Appl_No=213535) |
| [ ] | app09 | dev | On what date did the FDA first approve Promacta? | NDA022291 was approved 2008-11-20. | date | [1](https://www.accessdata.fda.gov/scripts/cder/daf/index.cfm?event=overview.process&ApplNo=022291) |
| [ ] | app04 | test | On what date did the FDA first approve Trodelvy? | BLA761115 was approved 2020-04-22. | date | [1](https://www.accessdata.fda.gov/scripts/cder/daf/index.cfm?event=overview.process&ApplNo=761115) |
| [ ] | ind03 | dev | Is Trelegy Ellipta FDA-approved for asthma? | Yes: the current Trelegy Ellipta label lists it. | yes_no | [1](https://dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid=b5a81d5a-4648-4c7a-951d-33c014a63c7e) |
| [ ] | ind09 | test | Is Imdelltra FDA-approved for limited-stage small cell lung cancer? | No: the current Imdelltra label does not list it. | yes_no | [1](https://dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid=1e7b6163-5d83-42ea-82c9-cf7620cdc782) |
| [ ] | trl04 | dev | What is the current overall status and primary completion date of NCT04455750? | NCT04455750 is ACTIVE_NOT_RECRUITING, primary completion 2027-03-15. | trial_status | [1](https://clinicaltrials.gov/study/NCT04455750) |
| [ ] | trl05 | test | Where does the Erleada study NCT02531516 stand, and when should it read out? | NCT02531516 is ACTIVE_NOT_RECRUITING, primary completion 2027-03-01. | trial_status | [1](https://clinicaltrials.gov/study/NCT02531516) |
| [ ] | win02 | test | Any Imdelltra Phase 3 readouts coming in the next 12 months? | 2 trials: NCT06211036, NCT07778316. | nct_set | [1](https://clinicaltrials.gov/study/NCT07778316) [2](https://clinicaltrials.gov/study/NCT06211036) |
| [ ] | win01 | test | Which Phase 3 trials of risdiplam that are recruiting, active or not yet recruiting have a primary completion date between 2026-11-04 and 2027-11-04? List their NCT numbers. | 1 trial: NCT05067790. | nct_set | [1](https://clinicaltrials.gov/study/NCT05067790) |
| [ ] | tk15 | test | What were Xtandi's 2025 end-market sales, as reported in Royalty Pharma's 10-K? | $6,297 million in 2025 end-market sales. | money | [1](https://www.sec.gov/Archives/edgar/data/1802768/000180276826000003/rprx-20251231.htm) |
| [ ] | tk10 | test | How long will Royalty Pharma keep collecting on Erleada? | Estimated royalty duration 2032. | contains_all | [1](https://www.sec.gov/Archives/edgar/data/1802768/000180276826000003/rprx-20251231.htm) |
| [ ] | cnf01 | dev | When do Trikafta's patents expire? Compare Royalty Pharma's 10-K with the FDA Orange Book. | 10-K: We estimate expected Trikafta patent expiration in 2037 Orange Book: the last-expiring Trikafta patent expires 2038-07-17 | contains_all | [1](https://www.accessdata.fda.gov/scripts/cder/ob/results_product.cfm?Appl_Type=N&Appl_No=212273) [2](https://www.sec.gov/Archives/edgar/data/1802768/000180276826000003/rprx-20251231.htm) |
| [ ] | cnf02 | test | Royalty Pharma ties its cystic fibrosis royalty duration to Alyftrek's patents. What does its 10-K estimate, and when does Alyftrek's last Orange Book patent expire? | 10-K: We estimate royalty duration of 2039-2041 due to expected Alyftrek patent expiration Orange Book: the last-expiring Alyftrek patent expires 2043-01-10 | contains_all | [1](https://www.accessdata.fda.gov/scripts/cder/ob/results_product.cfm?Appl_Type=N&Appl_No=218730) [2](https://www.sec.gov/Archives/edgar/data/1802768/000180276826000003/rprx-20251231.htm) |
| [ ] | bla03 | test | When does the last-expiring FDA Orange Book patent listed for Trodelvy expire? | Trodelvy is a biologic (BLA); the Orange Book lists no patents for it. | none_listed | Orange Book has no listing |
| [ ] | bla01 | dev | When does the last-expiring FDA Orange Book patent listed for Tremfya expire? | Tremfya is a biologic (BLA); the Orange Book lists no patents for it. | none_listed | Orange Book has no listing |
| [ ] | oos10 | test | When did the European Commission approve Voranigo? | NOT IN SOURCES | decline | none (should decline) |
| [ ] | oos07 | test | What will Royalty Pharma's total revenue be in 2027? | NOT IN SOURCES | decline | none (should decline) |

## All questions

### patent

| ID | Split | Style | Question | Gold | Check |
|---|---|---|---|---|---|
| pat01 | test | direct | When does the last-expiring FDA Orange Book patent listed for Xtandi expire? | Patent 12161628 expires 2037-02-23. | date |
| pat02 | test | plain | When does Trikafta lose its last Orange Book patent protection? | Patent 12350262 expires 2038-07-17. | date |
| pat03 | test | direct | When does the last-expiring FDA Orange Book patent listed for Alyftrek expire? | Patent 12186306 expires 2043-01-10. | date |
| pat04 | dev | direct | When does the last-expiring FDA Orange Book patent listed for Kalydeco expire? | Patent 8883206*PED expires 2033-08-27. | date |
| pat05 | test | plain | When does Imbruvica lose its last Orange Book patent protection? | Patent 12201690 expires 2039-06-14. | date |
| pat06 | test | direct | When does the last-expiring FDA Orange Book patent listed for Evrysdi expire? | Patent 12122789 expires 2041-04-15. | date |
| pat07 | test | direct | When does the last-expiring FDA Orange Book patent listed for Erleada expire? | Patent 11723898 expires 2041-01-14. | date |
| pat08 | dev | plain | When does Voranigo lose its last Orange Book patent protection? | Patent 11345677 expires 2039-01-16. | date |
| pat09 | test | direct | When does the last-expiring FDA Orange Book patent listed for Cabometyx expire? | Patent 9724342 expires 2033-07-09. | date |
| pat10 | test | direct | When does the last-expiring FDA Orange Book patent listed for Spinraza expire? | Patent 12013403 expires 2036-03-04. | date |
| pat11 | dev | plain | When does Amvuttra lose its last Orange Book patent protection? | Patent 11286486 expires 2036-07-28. | date |
| pat12 | dev | direct | When does the last-expiring FDA Orange Book patent listed for Farxiga expire? | Patent 12409186*PED expires 2041-10-01. | date |

### exclusivity

| ID | Split | Style | Question | Gold | Check |
|---|---|---|---|---|---|
| exc01 | test | direct | What is the latest-expiring FDA regulatory exclusivity listed for Trikafta in the Orange Book, and when does it expire? | Exclusivity ODE-543 expires 2033-03-27. | date |
| exc02 | test | plain | How long does Alyftrek's FDA exclusivity run? | Exclusivity ODE-536 expires 2033-03-27. | date |
| exc03 | dev | direct | What is the latest-expiring FDA regulatory exclusivity listed for Voranigo in the Orange Book, and when does it expire? | Exclusivity ODE-491 expires 2031-08-06. | date |
| exc04 | test | direct | What is the latest-expiring FDA regulatory exclusivity listed for Myqorzo in the Orange Book, and when does it expire? | Exclusivity NCE expires 2030-12-19. | date |
| exc05 | test | plain | How long does Orladeyo's FDA exclusivity run? | Exclusivity PED expires 2029-06-11. | date |
| exc06 | dev | direct | What is the latest-expiring FDA regulatory exclusivity listed for Evrysdi in the Orange Book, and when does it expire? | Exclusivity ODE-400 expires 2029-05-27. | date |

### approval

| ID | Split | Style | Question | Gold | Check |
|---|---|---|---|---|---|
| app01 | test | direct | On what date did the FDA first approve Tysabri? | BLA125104 was approved 2004-11-23. | date |
| app02 | test | plain | When did Xtandi first get the FDA nod? | NDA203415 was approved 2012-08-31. | date |
| app03 | dev | direct | On what date did the FDA first approve Tremfya? | BLA761061 was approved 2017-07-13. | date |
| app04 | test | direct | On what date did the FDA first approve Trodelvy? | BLA761115 was approved 2020-04-22. | date |
| app05 | test | plain | When did Imdelltra first get the FDA nod? | BLA761344 was approved 2024-05-16. | date |
| app06 | dev | direct | On what date did the FDA first approve Voranigo? | NDA218784 was approved 2024-08-06. | date |
| app07 | test | direct | On what date did the FDA first approve Myqorzo? | NDA219083 was approved 2025-12-19. | date |
| app08 | test | plain | When did Kalydeco first get the FDA nod? | NDA203188 was approved 2012-01-31. | date |
| app09 | dev | direct | On what date did the FDA first approve Promacta? | NDA022291 was approved 2008-11-20. | date |
| app10 | test | direct | On what date did the FDA first approve Spinraza? | NDA209531 was approved 2016-12-23. | date |

### indication

| ID | Split | Style | Question | Gold | Check |
|---|---|---|---|---|---|
| ind01 | test | direct | Is Tremfya FDA-approved for Crohn's disease? | Yes: the current Tremfya label lists it. | yes_no |
| ind02 | test | plain | Can Xtandi be prescribed on-label for non-metastatic castration-sensitive prostate cancer with biochemical recurrence? | Yes: the current Xtandi label lists it. | yes_no |
| ind03 | dev | direct | Is Trelegy Ellipta FDA-approved for asthma? | Yes: the current Trelegy Ellipta label lists it. | yes_no |
| ind04 | test | direct | Is Trodelvy FDA-approved for triple-negative breast cancer? | Yes: the current Trodelvy label lists it. | yes_no |
| ind05 | dev | plain | Can Amvuttra be prescribed on-label for the cardiomyopathy of transthyretin-mediated amyloidosis? | Yes: the current Amvuttra label lists it. | yes_no |
| ind06 | test | direct | Is Tysabri FDA-approved for ulcerative colitis? | No: the current Tysabri label does not list it. | yes_no |
| ind07 | test | direct | Is Spinraza FDA-approved for amyotrophic lateral sclerosis (ALS)? | No: the current Spinraza label does not list it. | yes_no |
| ind08 | test | plain | Can Erleada be prescribed on-label for non-metastatic castration-sensitive prostate cancer? | No: the current Erleada label does not list it. | yes_no |
| ind09 | test | direct | Is Imdelltra FDA-approved for limited-stage small cell lung cancer? | No: the current Imdelltra label does not list it. | yes_no |
| ind10 | dev | direct | Is Voranigo FDA-approved for glioblastoma? | No: the current Voranigo label does not list it. | yes_no |

### trial

| ID | Split | Style | Question | Gold | Check |
|---|---|---|---|---|---|
| trl01 | test | direct | What is the current overall status and primary completion date of NCT06211036? | NCT06211036 is ACTIVE_NOT_RECRUITING, primary completion 2027-07-05. | trial_status |
| trl02 | test | plain | Where does the Evrysdi study NCT05067790 stand, and when should it read out? | NCT05067790 is ACTIVE_NOT_RECRUITING, primary completion 2027-09-24. | trial_status |
| trl03 | test | direct | What is the current overall status and primary completion date of NCT05347095? | NCT05347095 is ACTIVE_NOT_RECRUITING, primary completion 2027-03-25. | trial_status |
| trl04 | dev | direct | What is the current overall status and primary completion date of NCT04455750? | NCT04455750 is ACTIVE_NOT_RECRUITING, primary completion 2027-03-15. | trial_status |
| trl05 | test | plain | Where does the Erleada study NCT02531516 stand, and when should it read out? | NCT02531516 is ACTIVE_NOT_RECRUITING, primary completion 2027-03-01. | trial_status |
| trl06 | dev | direct | What is the current overall status and primary completion date of NCT05254743? | NCT05254743 is RECRUITING, primary completion 2027-10. | trial_status |

### readouts

| ID | Split | Style | Question | Gold | Check |
|---|---|---|---|---|---|
| win01 | test | direct | Which Phase 3 trials of risdiplam that are recruiting, active or not yet recruiting have a primary completion date between 2026-11-04 and 2027-11-04? List their NCT numbers. | 1 trial: NCT05067790. | nct_set |
| win02 | test | plain | Any Imdelltra Phase 3 readouts coming in the next 12 months? | 2 trials: NCT06211036, NCT07778316. | nct_set |
| win03 | test | direct | Which Phase 3 trials of nusinersen that are recruiting, active or not yet recruiting have a primary completion date between 2026-11-04 and 2027-11-04? List their NCT numbers. | 2 trials: NCT05067790, NCT06971094. | nct_set |
| win04 | dev | direct | Which Phase 3 trials of apalutamide that are recruiting, active or not yet recruiting have a primary completion date between 2026-11-04 and 2027-11-04? List their NCT numbers. | 4 trials: NCT02531516, NCT03767244, NCT03903835, NCT06274047. | nct_set |
| win05 | test | plain | Any Tremfya Phase 3 readouts coming in the next 12 months? | 5 trials: NCT05347095, NCT05535738, NCT05923073, NCT06260163, NCT06408935. | nct_set |
| win06 | dev | direct | Which Phase 3 trials of sacituzumab govitecan that are recruiting, active or not yet recruiting have a primary completion date between 2026-11-04 and 2027-11-04? List their NCT numbers. | 5 trials: NCT04595565, NCT04639986, NCT05382286, NCT05609968, NCT05633654. | nct_set |

### tenk

| ID | Split | Style | Question | Gold | Check |
|---|---|---|---|---|---|
| tk01 | dev | plain | What cut of Xtandi sales does Royalty Pharma get? | Slightly less than 4% royalty (10-K Portfolio Summary). | judge |
| tk02 | test | direct | What royalty rate does Royalty Pharma receive on Tremfya, according to its latest 10-K? | ~4% royalty (10-K Portfolio Summary). | judge |
| tk03 | test | direct | What royalty rate does Royalty Pharma receive on Voranigo, according to its latest 10-K? | Tiered royalty of 15% on first $1 billion of U.S. sales, down to 12% on U.S. sales >$1 billion (10-K Portfolio Summary). | judge |
| tk04 | test | plain | What cut of Trodelvy sales does Royalty Pharma get? | Tiered royalty of 4.15% on first $2 billion, down to 1.75% on sales >$6 billion (10-K Portfolio Summary). | judge |
| tk05 | dev | direct | What royalty rate does Royalty Pharma receive on Tysabri, according to its latest 10-K? | Tiered payments of 18% on first $2 billion and 25% on sales >$2 billion (10-K Portfolio Summary). | judge |
| tk06 | test | direct | What royalty rate does Royalty Pharma receive on Imdelltra, according to its latest 10-K? | ~7% royalty with royalty sharing on sales >$1.5 billion (10-K Portfolio Summary). | judge |
| tk07 | test | plain | How long will Royalty Pharma keep collecting on Evrysdi? | Estimated royalty duration 2035-2036. | contains_all |
| tk08 | test | direct | What is Royalty Pharma's estimated royalty duration for Promacta? | Estimated royalty duration 2025-2028. | contains_all |
| tk09 | test | direct | What is Royalty Pharma's estimated royalty duration for Tremfya? | Estimated royalty duration 2031-2032. | contains_all |
| tk10 | test | plain | How long will Royalty Pharma keep collecting on Erleada? | Estimated royalty duration 2032. | contains_all |
| tk11 | test | direct | What were Royalty Pharma's 2025 Portfolio Receipts from Tysabri? | $250 million in 2025 Portfolio Receipts. | money |
| tk12 | test | direct | What were Royalty Pharma's 2025 Portfolio Receipts from Trelegy? | $332 million in 2025 Portfolio Receipts. | money |
| tk13 | dev | plain | How much royalty cash did Spinraza bring Royalty Pharma in 2025? | $52 million in 2025 Portfolio Receipts. | money |
| tk14 | dev | direct | What were Royalty Pharma's 2025 Portfolio Receipts from Imbruvica? | $170 million in 2025 Portfolio Receipts. | money |
| tk15 | test | direct | What were Xtandi's 2025 end-market sales, as reported in Royalty Pharma's 10-K? | $6,297 million in 2025 end-market sales. | money |
| tk16 | test | plain | How big was Tremfya in 2025 sales terms? | $5,155 million in 2025 end-market sales. | money |
| tk17 | test | direct | What were Imbruvica's 2025 end-market sales, as reported in Royalty Pharma's 10-K? | $3,979 million in 2025 end-market sales. | money |
| tk18 | test | direct | In what year did Royalty Pharma acquire its royalty on Tysabri? | Acquired in 2017. | contains_all |
| tk19 | test | plain | When did Royalty Pharma buy into Trodelvy? | Acquired in 2018. | contains_all |
| tk20 | dev | direct | What royalty rate does Royalty Pharma expect on Alyftrek, and what is in dispute about it? | Royalty Pharma believes deutivacaftor is royalty-bearing, giving a blended royalty of about 8% Vertex says deutivacaftor is not royalty-bearing, which would give about 4% | judge |
| tk21 | dev | direct | When does Royalty Pharma's royalty interest in Spinraza revert to Ionis? | It reverts to Ionis after Royalty Pharma receives aggregate Spinraza royalties of $475 million or $550 million, depending on the timing and occurrence of certain events | judge |
| tk22 | dev | direct | Until when does Royalty Pharma receive royalties on U.S. sales of cabozantinib products? | Through September 2026 for U.S. sales; non-U.S. markets through the full term of the royalty | contains_all |
| tk23 | dev | direct | When did the European Commission approve Alyftrek, according to Royalty Pharma's 10-K? | July 2025 | contains_all |

### conflict

| ID | Split | Style | Question | Gold | Check |
|---|---|---|---|---|---|
| cnf01 | dev | direct | When do Trikafta's patents expire? Compare Royalty Pharma's 10-K with the FDA Orange Book. | 10-K: We estimate expected Trikafta patent expiration in 2037 Orange Book: the last-expiring Trikafta patent expires 2038-07-17 | contains_all |
| cnf02 | test | direct | Royalty Pharma ties its cystic fibrosis royalty duration to Alyftrek's patents. What does its 10-K estimate, and when does Alyftrek's last Orange Book patent expire? | 10-K: We estimate royalty duration of 2039-2041 due to expected Alyftrek patent expiration Orange Book: the last-expiring Alyftrek patent expires 2043-01-10 | contains_all |

### biologic

| ID | Split | Style | Question | Gold | Check |
|---|---|---|---|---|---|
| bla01 | dev | direct | When does the last-expiring FDA Orange Book patent listed for Tremfya expire? | Tremfya is a biologic (BLA); the Orange Book lists no patents for it. | none_listed |
| bla02 | test | plain | When do Tysabri's Orange Book patents run out? | Tysabri is a biologic (BLA); the Orange Book lists no patents for it. | none_listed |
| bla03 | test | direct | When does the last-expiring FDA Orange Book patent listed for Trodelvy expire? | Trodelvy is a biologic (BLA); the Orange Book lists no patents for it. | none_listed |

### out_of_scope

| ID | Split | Style | Question | Gold | Check |
|---|---|---|---|---|---|
| oos01 | dev | direct | What is the U.S. list price (WAC) for a month of Trikafta? | NOT IN SOURCES | decline |
| oos02 | test | plain | What's the analyst consensus price target for Royalty Pharma stock? | NOT IN SOURCES | decline |
| oos03 | test | direct | What royalty rate does Royalty Pharma receive on Ozempic? | NOT IN SOURCES | decline |
| oos04 | test | direct | What gross margin does Vertex earn on Trikafta? | NOT IN SOURCES | decline |
| oos05 | test | direct | When does the EU supplementary protection certificate for Trikafta expire? | NOT IN SOURCES | decline |
| oos06 | test | plain | What are analysts forecasting for Voranigo's peak sales? | NOT IN SOURCES | decline |
| oos07 | test | direct | What will Royalty Pharma's total revenue be in 2027? | NOT IN SOURCES | decline |
| oos08 | dev | direct | How much did Merck report in worldwide Keytruda sales for 2025? | NOT IN SOURCES | decline |
| oos09 | test | direct | How much did Medicare Part D spend on Tremfya in 2024? | NOT IN SOURCES | decline |
| oos10 | test | direct | When did the European Commission approve Voranigo? | NOT IN SOURCES | decline |
| oos11 | dev | plain | What's Imbruvica's U.S. market share in CLL right now? | NOT IN SOURCES | decline |
| oos12 | test | direct | Who is the chief executive of Servier? | NOT IN SOURCES | decline |
| oos13 | test | direct | How many units of Xtandi did Astellas ship in the third quarter of 2026? | NOT IN SOURCES | decline |
| oos14 | dev | direct | What did Royalty Pharma's board decide at its December 2026 meeting? | NOT IN SOURCES | decline |
| oos15 | dev | plain | What does a course of Evrysdi cost in Japan? | NOT IN SOURCES | decline |
