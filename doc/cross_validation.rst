Time Gap Split Cross Validation
===============================

TimeGapSplit provides 3 parameters to really reproduce your production implementation in your cross-validation schema.

It uses a column parameter to specify the date used to split.
In case you have multiple rows per day (or per timestamp) this will make sure those rows in the same day
won't get split in different folds. This is a drawback of the original scikit-learn TimeSerieSplit that we fixed here.

Below a picture showing the all CVs with a gap.

.. image:: _static/timegapsplit.png
   :align: center

Check the notebook for example and graph.


train_duration
**************
In the graph above in blue.

That represents the length of your rolling window used for in your feature generation.

validation_duration
*******************
In the graph above in red.


gap_duration
*******************
In the graph above in white between train and validation

This gap is a simulation of the fact that in production you are dropping the last forward looking window available,
i.e. there is alway 1 forward looking window between:

- the last day of your training sample on which you can create the target (which needs a forward looking window)
- and the first production sample scoring.



