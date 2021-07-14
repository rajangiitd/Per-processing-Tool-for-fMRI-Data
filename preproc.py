import numpy as np
from scipy.fftpack import fft, ifft
import argparse
import sys
import nibabel as nb
#----------------------------------------------------------------------------------------------------------------------#
######### slice time corection for one time series
def sliceTimeCorrectionForOneTimeSeries(y, sliceAcquisitionTime, TR, targetTime):
    series1=[]
    series2=[]
    factor=0
    if targetTime < sliceAcquisitionTime:
        factor = (targetTime + TR - sliceAcquisitionTime) / TR
        series1 = y[:-2]
        series2 = y[1:-1]
        my_y = np.concatenate(([y[0]], series2 * factor + (1 - factor) * series1, [y[-1]]))
        return my_y
    elif targetTime > sliceAcquisitionTime:
        factor = (targetTime - sliceAcquisitionTime) / TR
        series1 = y[1:-1]
        series2 = y[2:]
        my_y = np.concatenate( ([y[0]],  series2*factor + (1-factor) *series1, [y[-1]]) )
        return my_y

################## slice time correction function
def sliceTimeCorrection(fMRIData, TR, targetTime, sliceTimeAcquisitionFile, outputFile):
    text_file_for_output = open(outputFile+ '.txt', 'w')
    Verdict = "SLICE TIME CORRECTION "
    if targetTime > TR*1000 or targetTime < 0:
        text_file_for_output.write( Verdict +"FAILURE")
        return fMRIData
    else:
        TR = TR*1000
        sliceAcquisitionTimes = np.loadtxt(sliceTimeAcquisitionFile)
        datashape = np.shape(fMRIData)
        OutputImage = np.zeros(datashape)

        for slice_number in range(datashape[2]):
            if sliceAcquisitionTimes[slice_number] - TR > 0 or sliceAcquisitionTimes[slice_number] < 0:
                text_file_for_output.write(Verdict+"FAILURE")
                return fMRIData
            for x in range(datashape[0]):
                for y in range(datashape[1]):
                    OutputImage[x][y][slice_number] = sliceTimeCorrectionForOneTimeSeries(fMRIData[x][y][slice_number], sliceAcquisitionTimes[slice_number], TR, targetTime)

        text_file_for_output.write(Verdict+"SUCCESS")
        return OutputImage

#----------------------------------------------------------------------------------------------------------------------#

def temporalFiltering(low , high, TR, fMRIData):
    Higher_cutoff_frequency= 1/min([high,low])
    Lower_cutoff_frequency = 1/max([high,low])
    datashape= np.shape(fMRIData)
    X=datashape[0]
    Y=datashape[1]
    Z=datashape[2]
    final_data = np.zeros(datashape)
    for x in range(X):
        for y in range(Y):
            for z in range(Z):
                Time_series_for_a_voxel = fMRIData[x][y][z]
                n = len(Time_series_for_a_voxel)
                frequencies = np.fft.fftfreq(n, d= TR)
                YFourier = fft(Time_series_for_a_voxel)
                YFourier[np.abs(frequencies) > Higher_cutoff_frequency] = 0
                #YFourier[np.abs(frequencies) < Lower_cutoff_frequency] = 0
                Band_pass_filtered_time_series = ifft(YFourier)
                final_data[x][y][z][:] = Band_pass_filtered_time_series
    return final_data

#----------------------------------------------------------------------------------------------------------------------#
########### preparation or formation of gaussian kernel ###############
def Form_Gaussian_Kernel(k_size, fwhm, voxel_size):
    x = np.arange(-k_size, k_size+1)
    sigma_value = fwhm / (np.sqrt(8 * np.log(2)))
    sx = sigma_value/float(voxel_size)
    y = np.exp(-(x ** 2) / (2 * (sx ** 2)))
    ## returning kernel
    return y/sum(y)

def useKernel( kernel, sequence):
    Length_of_kernel = np.size(kernel)
    Length_of_sequence = np.size(sequence)
    paddedSequence = np.concatenate((np.flip(sequence), sequence, np.flip(sequence)))
    Result = np.zeros(Length_of_sequence)

    for startingPointer in range(0, Length_of_sequence):
        Result[startingPointer] = np.sum(np.multiply(kernel, paddedSequence[startingPointer + Length_of_sequence - Length_of_kernel//2 : startingPointer + 1 + Length_of_kernel//2 + Length_of_sequence]))
    return Result

def spatialSmoothing(fMRIData, fwhm, voxelDimensions):
    data_shape = np.shape(fMRIData)
    X= data_shape[0]
    Y= data_shape[1]
    Z= data_shape[2]
    T= data_shape[3]
    #print(voxelDimensions)
    #voxelDimensions = headr['pixdim']
    Output_fMRIData = np.zeros(np.shape(fMRIData))
    for t in range(T):
        # applying kernel along x
        kernelX = Form_Gaussian_Kernel(X, fwhm, voxelDimensions[0])
        OutX = np.zeros(np.shape(fMRIData)[:-1])
        for y in range(Y):
            for z in range(Z):
                OutX[:, y, z] = useKernel( kernelX,fMRIData[:, y, z, t])

        # applying kernel along y
        OutY = np.zeros(np.shape(fMRIData)[:-1])
        kernelY = Form_Gaussian_Kernel(Y, fwhm, voxelDimensions[1])
        for x in range(X):
            for z in range(Z):
                OutY[x, :, z] = useKernel(kernelY,OutX[x, :, z])

        # applying kernel along z
        OutZ = np.zeros(np.shape(fMRIData)[:-1])
        kernelZ = Form_Gaussian_Kernel(Z, fwhm, voxelDimensions[2])
        for x in range(X):
            for y in range(Y):
                OutZ[x, y, :] = useKernel( kernelZ, OutY[x, y, :])
        Output_fMRIData[...,t] = OutZ
    return Output_fMRIData

#----------------------------------------------------------------------------------------------------------------------#

Input_parser = argparse.ArgumentParser()
Input_parser.add_argument("-i", "--inputFile", required=True)
Input_parser.add_argument('-o', '--outputFile', required=True)
Input_parser.add_argument('-tc', '--listSliceTimeCorrection', nargs='+', default=[0, 0])
Input_parser.add_argument('-tf', '--listTemporalFiltering', nargs='+', default=[0, 0])
Input_parser.add_argument('-sm', '--fwhm', default=0)
args = Input_parser.parse_args()

#----------------------------------------------------------------------------------------------------------------------#
outputFile = str(args.outputFile)
inputFile = str(args.inputFile)

sliceTimeAcquisitionFile = args.listSliceTimeCorrection[1]
targetTime = float(args.listSliceTimeCorrection[0])

high = float(args.listTemporalFiltering[0])
low = float(args.listTemporalFiltering[1])

fwhm = float(args.fwhm)
#----------------------------------------------------------------------------------------------------------------------#
img = nb.load(inputFile)
fMRIData = img.get_fdata()

Not_A_Empty_Input = False

TR = img.header.get_zooms()[3]
voxelDimensions = img.header.get_zooms()[:3]

number_of_sys_arguments_given = len(sys.argv)

#----------------------------------------------------------------------------------------------------------------------#
for argIndex in range(number_of_sys_arguments_given):
    arg = sys.argv[argIndex]
    if arg == '--listSliceTimeCorrection' or arg == '-tc' :
        Not_A_Empty_Input = True
        fMRIData = sliceTimeCorrection(fMRIData, TR, targetTime, sliceTimeAcquisitionFile, outputFile)
    elif arg == '--listTemporalFiltering' or arg == '-tf' :
        Not_A_Empty_Input = True
        fMRIData = temporalFiltering(low,high, TR,fMRIData)
    elif arg == '-sm' or arg == '--fwhm':
        Not_A_Empty_Input = True
        fMRIData = spatialSmoothing(fMRIData, fwhm, voxelDimensions)

#----------------------------------------------------------------------------------------------------------------------#
outputNIIFileName = outputFile + '.nii.gz'
if Not_A_Empty_Input:
    fMRIData = np.array(fMRIData, dtype = img.header.get_data_dtype())
    outputImg = nb.Nifti1Image(fMRIData, np.eye(4), header= img.header)
    nb.save(outputImg, outputNIIFileName)

#----------------------------------------------------------------------------------------------------------------------#
