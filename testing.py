import argparse
import nibabel as nb
import numpy as np
if __name__=='__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("-i", "--input", nargs=2, help="input files", required=True)
  args = parser.parse_args()
  img1 = nb.load(args.input[0])
  img2 = nb.load(args.input[1])
  data1 = np.array(img1.get_data())
  data2 = np.array(img2.get_data())
  diff = np.absolute(np.subtract(data1,data2))
  print("For input1, min: ",np.min(data1), " max: ", np.max(data1), " mean: ", np.mean(data1)," Std. Deviation: ", np.std(data1),"\n")
  print("For input2, min: ",np.min(data2), " max: ", np.max(data2), " mean: ", np.mean(data2)," Std. Deviation: ", np.std(data2),"\n")
  m = np.amin(diff)
  M = np.amax(diff)
  mean = np.mean(diff)
  std = np.std(diff)
  print("In absolute difference/error array")
  print("min: ",m," max: ",M," mean: ",mean, " Std. Deviation: ", std)
  below_mean=0
  below_mean_plus_std=0
  Shape= np.shape(diff)
  X= Shape[0]
  Y= Shape[1]
  Z= Shape[2]
  T= Shape[3]
  D= np.array(diff).tolist()
  Arr = [0 for i in range(11)]
  for x in range(X):
    for y in range(Y):
      for z in range(Z):
        for t in range(T):
          if M!=0:
            Arr[ int(((D[x][y][z][t]/M)*10)//1)]+=1
          else:
            Arr[0]+=1
          if(D[x][y][z][t]<mean):
            below_mean+=1
            below_mean_plus_std+=1
          elif(D[x][y][z][t]<mean+std):
            below_mean_plus_std+=1
  S = sum(Arr)
  print("Total data points in Array : "+str(S))
  print(str((below_mean/S)*100)+" % errors are below mean error = "+str(mean))
  print(str((below_mean_plus_std / S) * 100) + " % errors are below mean+std of error = " + str(mean+std))
  print("Counts from x% of Max_error to x+10%(exclusive) of Max_error")
  print( str(Arr))
  for i in range(11):
    Arr[i]= round((Arr[i]*100)/S,3)
  print("In terms of percentage ")
  print(str(Arr))

