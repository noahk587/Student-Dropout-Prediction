# This is a function that takes a vector containing the actual response
# values and a vector containing the predicted values of the response values.
# The function returns a list containg a ROC curve and the AUC value.
# @parm   trueLabels      A vector that contains the acutal response values
# @parm   predictedLabel  A vector that contains the predicted response values
# @parm   modelName       A string that has the name of the predictive model
# @return rocPlot         The plot of the ROC curve
# @return aucValue        The value of the AUC

ROCplot <- function(trueLabels, predictedLabel, modelName){
  rocPlot <- ggplot(data = NULL, aes(m = predictedLabel, d = trueLabels)) + 
    geom_roc(n.cuts = 0, color = "deepskyblue3") + labs(x = "False Positive Rate",
      y = "True Positve Rate") 
  aucValue <- calc_auc(rocPlot)
  aucValue <- round(aucValue[1, 3], 4)
  rocPlot <- rocPlot + ggtitle(paste0("ROC Curve for ", modelName,
                                      " (AUC = ", aucValue, ")"))
  
  return(list(rocPlot, aucValue))
}