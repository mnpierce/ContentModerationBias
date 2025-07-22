# Content Moderation: Using Machine Learning to Classify Toxic Comments with Bias Analysis
CSE3000 - Final Project  
Course Instructor: Rye Howard-Stone  
Group Members: Alex Reilly, Andrew Khan, Matt Pierce, Wes Deal  

## About the Data Source: Civil Comments (Platform)

- Commenting plugin for independent news sites 
- Shut down 2017 due to lack of resources, made all data public 
- Required users to rate the civility of three other random comments before making their own 
- Worked great when an article went viral, more comments more reviews 

## Dataset Labels

### Labels by Paid Annotators (Figure Eight crowd rating platform 2018, 10-thousands per comment) 

- “target” (fraction of annotators labelling as toxic or very toxic) 
- \>= 0.5 - TOXIC 
- < 0.5 - NOT TOXIC 

#### Toxicity type 

- severe_toxicity, obscene, threat, Insult, identity_attack, sexual_explicit 

#### Demographics (used to analyze bias)

- Male, female, transgender, other_gender, heterosexual, homosexual_gay_or_lesbian, bisexual, other_sexual_orientation, christian, jewish, muslim, hindu, buddhist, atheist, other_religion, black, white, asian, latino, other_race_or_ethnicity, physical_disability, intellectual_or_learning_disability, psychiatric_or_mental_illness, other_disability 

### Labels by Platform users 

- “rating” (calculated how?) ... cant really use this if we dont know how its calculated 
- Interactions (funny, wow, sad, likes, disagree) 

## Dataset Modifications
### Missing Values
Because the primary goal of this project is to analyze bias, we must have demographic labels associated with our data. Removing samples that were missing demographic labels reduced the number of samples from ~1.8 million to ~405 thousand.

### Sample size
Given that real people are labelling the "target" label, we should make sure that there are a reasonable amount of annotators. It is important to make sure this label is stable as it is the target label for our model. As a result, we removed any samples with a toxicity_annotator_count < 10. This further reduced the sample size down to ~108 thousand.

Under this topic, one limitation of our dataset is that the number of identity annotators (demographic labels) is often as low as 4. This could lead to some inconsistency in demographic scores, but should still serve sufficient to help us analyze bias in the model.

Note: We will first try to train with only data with attached demographic labels, but if results are bad we can try the rest of the data without demographic labels.

## Full run of baseline models and transformer model
0. Download all required libraries listed in `requirements.txt` (RECOMMENDED: Python 3.12)
1. Start with `full_train.csv`
2. Run `data_trim.py` to create `trimmed_train.csv`
    - `trimmed_train.csv` is all you need to run `trainer_bert.py`
3. Run `tokenize_text.py` to create `tokenized_train.csv`
    - This csv is also provided to you so you can skip these steps 
4. Run `baseline_models.py` to train and evaluate Log. Reg., NB models

## Jupyter Notebook walkthrough with results visualization and bias data
1. Run `Content_Moderation.ipynb`

## NPM Setup
1. Download Node.js
2. `npm install -D tailwindcss postcss autoprefixer` (configuration already done in repository)
3. `npm install`
4. `npm install --save-dev tsx`
5. `npm install -D @tailwindcss/postcss`
6. `npm run dev`

Note: also run `api.py` simultaneously to start web app (requires `pip install Flask flask-cors`)