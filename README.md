Domain: Support

## How to run

Requirements: Python 3.10 or above.

Install the required package:

```bash
python -m pip install -r requirements.txt
```

Run the following commands from the project root folder:

```bash
python generate_data.py
python part1.py
python part2.py
python part3.py
python part4.py
```

All results and CSV files are saved in the `outputs/` folder.

## Part 1 findings

- The dataset contains 800 records.
- The average resolution time is 8.9382 minutes, with a standard deviation of 9.1723 minutes.
- The 95% confidence interval for the mean is 8.3026 to 9.5738 minutes.
- The hypothesis test rejects the assumption that the population mean is 8 minutes.
- The simple regression model has a slope of 0.6701 and an intercept of -3.6598.
- The model R-squared is 0.5196, with an RMSE of 6.3537.
- Gradient descent reached the same slope and intercept as the closed-form regression.
- Residual analysis showed greater variation at higher fitted values. The log-log model reduced the residual standard-deviation ratio from 3.7525 to 1.0396.
- Multiple regression achieved an R-squared of 0.7425 and an adjusted R-squared of 0.7403.

## Part 2 findings

- The data was divided into 560 training records and 240 testing records.
- The training set contained 119 positive cases, while the test set contained 51.
- Logistic regression training loss decreased from 0.6931 to 0.3704.
- X3 had the highest odds ratio (2.4629), followed by X2 (1.7341).
- At threshold 0.5, logistic regression achieved 80.42% accuracy and an F1-score of 0.5053.
- At threshold 0.3, recall improved to 0.7451 and the F1-score improved to 0.5714.
- KNN selected K=5 and achieved an F1-score of 0.3696.
- The cost-based threshold was 0.15. It produced a training cost of INR 12,150 and a test cost of INR 6,050.
- The test cost at threshold 0.5 was INR 11,800, while flagging nobody resulted in INR 20,400.

## Part 3 findings

- K-means clustering was performed for K values from 2 to 6.
- The elbow method selected K=3 for further analysis.
- Segment A contains 322 records, Segment B contains 255 records, and Segment C contains 223 records.
- Segment A has an outcome rate of 1.86%, Segment B has 23.53%, and Segment C has the highest outcome rate of 46.64%.
- Segment C should receive greater attention from the support team because it has the highest outcome rate.
- Hierarchical clustering produced 4 clusters using single linkage and 2 clusters using complete and average linkage at their respective largest-gap cuts.
- The first two principal components explain 75.93% of the total variation.
- Three principal components are required to explain at least 80% of the variation.

## Part 4 findings

- Root entropy was 0.7462 and root Gini impurity was 0.3347.
- Split S4 achieved the highest information gain and Gini gain.
- The decision stump produced a positive rate of 45.74% for the yes group and 8.87% for the no group.
- AdaBoost was implemented for two rounds. The first round error was 0.2500 and the second round error was 0.1667.
- The ensemble accuracy was 75%.
- The out-of-bag fraction was 0.3676.
- The MLP contains 13,002 parameters, the CNN contains 15,734 parameters, and the RNN contains 26,768 parameters.
- The fully connected layer contains 50,332,672 parameters, compared with 1,792 parameters for the convolutional layer.

## Model-choice memo

**To: Head of Customer Support Operations**

I recommend logistic regression as the first structured model to evaluate for Swiggy support operations.

On the test dataset, logistic regression achieved an F1-score of 0.5714, precision of 0.4634 and recall of 0.7451. It performed better overall than KNN (F1-score 0.3696), the decision stump (0.5532), hard voting (0.5344) and soft voting (0.5507).

For cost-based threshold selection, a threshold of 0.15 resulted in a test cost of INR 6,050, compared with INR 11,800 at a threshold of 0.5. Flagging nobody resulted in a cost of INR 20,400.

The decision stump achieved slightly higher recall (0.7647), so it may be useful when identifying as many positive cases as possible is the main priority. However, logistic regression provides the best balance of precision and recall among the evaluated models.

The dataset is synthetic, so the model should first be tested on real support data through a controlled pilot. A complex neural-network model is not necessary before evaluating this simpler structured model.

## Limitations

- The dataset is synthetic and may not represent actual Swiggy customer-support behaviour.
- Model performance may change when applied to real-world data.
- The selected thresholds depend on the assumed false-positive and false-negative costs.
- Customer behaviour and support requirements may change over time.
- The models should be tested and monitored before production deployment.