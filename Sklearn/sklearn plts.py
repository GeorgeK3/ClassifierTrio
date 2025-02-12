import matplotlib.pyplot as plt


def draw_plot(percentage_used,Algorithm_name,ac,f1,precision,recall):

    fig, axis = plt.subplots(2, 2)

    axis[0,0].set_title("                                                                                                                             "+Algorithm_name)
    axis[0,0].plot(percentage_used,ac,'ro-',label="Accuracy %")
    axis[0,0].set_ylabel("Accuracy %")
    axis[0,0].set_xlabel("Train set size %")

    axis[0,1].plot(percentage_used,f1,'go-',label="F1")
    axis[0,1].set_ylabel("F1")
    axis[0,1].set_xlabel("Train set size %")

    axis[1,0].plot(percentage_used,precision,'bo-',label="Precision")
    axis[1,0].set_ylabel("Precision")
    axis[1,0].set_xlabel("Train set size %")

    axis[1,1].plot(percentage_used,recall,'o-',label="Recall")
    axis[1,1].set_ylabel("Recall")
    axis[1,1].set_xlabel("Train set size %")

    plt.show()

#############################################################################################################
'''
DATA
'''
#############################################################################################################
percentage_used = [10,20,30,40,50]

sklearn_LogReg_ac = [75,76,78,79,79]
sklearn_LogReg_f1 = [0.74,0.76,0.77,0.79,0.79]
sklearn_LogReg_precision = [0.76,0.79,0.77,0.8,0.8]
sklearn_LogReg_recall = [0.73,0.72,0.77,0.78,0.78]

sklearn_NaiveBayes_ac=[67,70,75,80,80]
sklearn_NaiveBayes_f1=[0.66,0.68,0.73,0.8,0.78]
sklearn_NaiveBayes_precision=[0.69,0.74,0.8,0.79,0.84]
sklearn_NaiveBayes_recall=[0.63,0.63,0.67,0.82,0.73]

sklearn_RandomForest_ac=[71,77,80,84,86]
sklearn_RandomForest_f1=[0.73,0.76,0.8,0.83,0.86]
sklearn_RandomForest_precision=[0.7,0.78,0.8,0.85,0.86]
sklearn_RandomForest_recall=[0.76,0.74,0.79,0.81,0.85]

#############################################################################################################
'''
DRAW PLOTS
'''
#############################################################################################################
draw_plot(percentage_used,"sklearn Logistic Regression",sklearn_LogReg_ac,sklearn_LogReg_f1,sklearn_LogReg_precision,sklearn_LogReg_recall)
draw_plot(percentage_used,"sklearn Naive Bayes",sklearn_NaiveBayes_ac,sklearn_NaiveBayes_f1,sklearn_NaiveBayes_precision,sklearn_NaiveBayes_recall)
draw_plot(percentage_used,"sklearn Random Forest",sklearn_RandomForest_ac,sklearn_RandomForest_f1,sklearn_RandomForest_precision,sklearn_RandomForest_recall)