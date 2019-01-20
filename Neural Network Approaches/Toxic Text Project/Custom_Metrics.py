

def accuracy_multilabel(y_true,y_pred):
    cnt=0

    N=len(y_pred[0])
    if(N==0):
        return 0

    total=len(y_true)

    for i in range(total):
        cnt_temp=0
        for j in range(N):
            if(y_true[i,j]==y_pred[i,j]):
                cnt_temp+=1
        cnt+=cnt_temp

    cnt/=N
    acc=cnt/total

    return acc