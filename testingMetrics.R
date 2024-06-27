# This is a function that takes a confusion matrix
# as parameter and returns the accuracy, precision,
# and recall.
# @param  confusionMatrix A confusion matrix of a model
# @return accuracy        The accuracy of the model
# @return precision       The precison value of the model
# @return recall          The recall value of the model

testingMetrics <- function(confusionMatrix){
  accuracy <- (confusionMatrix[1,1] + confusionMatrix[2,2]) / (confusionMatrix[1,1] + confusionMatrix[2,2]
                                                               + confusionMatrix[1,2] + confusionMatrix[2,1])
  
  precision <- confusionMatrix[1,1] / ( confusionMatrix[1,1] + confusionMatrix[2,1])
  
  recall <- confusionMatrix[1,1] / ( confusionMatrix[1,1] + confusionMatrix[1,2] )
  
  return(list(accuracy, precision, recall))
}