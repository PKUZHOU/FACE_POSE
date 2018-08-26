import os
import json
data_path = "/datasets/tmp/nvme1/POSE_data/"
anno_file = open("multi_angle.txt",'w')
folds = os.listdir(data_path)
number = 0
for fold in folds:
    files = os.listdir(data_path+fold)
    for file in files:
        if(file.split(".")[-1] == 'json'):
            filename = data_path+fold+'/'+file.split('.')[0]+'.jpg'
            with open(data_path+fold+'/'+file,'r') as f:
                annotation = json.load(f)
            angles = annotation["rotation"]
            yaw = angles["yaw"]
            roll = angles["roll"]
            pitch = angles["pitch"]
            anno_file.write(filename +' '+str(yaw)+' '+str(roll)+' '+str(pitch)+'\n')
            number+=1
            if number%1000 == 0:
                print(number)
anno_file.close()