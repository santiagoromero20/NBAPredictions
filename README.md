# NBA Predictions

In this project I trained two models. The first one to estimate the players salaries for the next season, and the other one to classify wether if a player will be or not in the next All-Star Game (this is a competition where only the 15 best player of the season play).

## Motivation

On a previous project I have made all the collecting, cleaning and pre-processing of the data. Now is time to take it into action and use it to train some models. The big question was, Which answer should I try to answer? After a time of talking it with some fellows and teammates, I decided the ones I mentioned previously.

## Technologies and Teachings

There were a lot of key aspects and concepts from everyday ML projects that I learned from this one. Starting from the difference between a Regression and Classification Model, how both are differently measured (it loss functions...) and evaluated. 

More particularly, I trained Linear Regressors (Univariable and Multivariate) and a DecisionTree Regressor. Then, for the classification task I used a Logistic Regressor. All this introduced me to the most useful (in my personal opinion...) library on ML, Scikit-Learn.

Also, the importance of performing Feature Engineer, and how to do it easily with the library mentioned earlier. Going more to the training part itself, the idea of applying a GridSearch to the models, in order to obtain the best hyperparameters possibles. This was firstly done manually, by this I mean coding it, then I did it with the awesome class from our good friends, Scikit-Learn.