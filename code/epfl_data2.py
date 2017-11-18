import cv2 as cv
import numpy as np
import csv

class Data:
    def __init__(self, rotations, bins):
        """
        Initalizes the data class by loading in the EPFL Multiview Car dataset
        Files have names of format: 'tripod_seq_#_#.jpg'
        :param rotations = M:
        :param bins = N:
         from the "Designing Deep Convolutional Neural Networks for Continuous
            Object Orientation Estimation" paper
        """
        
        info = open('data/tripod-seq.txt', 'r')
        data = info.readlines()
        self.width = 376
        self.height = 250
        self.num_seqs = 20
        self.pics_per_seq = [int(x) for x in data[1].split()]
        self.seq_indices = ['01', '02', '03', '04', '05', '06', '07', '08', '09', 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                            20]
        self.pics_per_circ = [int(x) for x in data[4].split()]
        self.seq_direction = [int(x) for x in data[6].split()]
        self.seq_frontal_index = [int(x) for x in data[5].split()]
        
        info.close()
        self.images = []
        self.targets = []
        self.samples = []

        self.rotations = rotations
        self.bins = bins
        self.labels = []
        self.binnies = []


        self._load_imgs()




    def _load_imgs(self):
        """
        Loads images and also generates targets at the same time
        """
        def _add_zeros(p):
            """
            Adds 0s in the front of single and double digit numbers because of the file name formatting
            :param p:
            :return 00x, 0y, or zzz:
            """
            if p < 10:
                p = '00' + str(p)
                return p
            elif p < 100:
                p = '0' + str(p)
                return p
            else:
                return p

        images = []
        image_seq = []
        targets = []
        labels = []
        seq_ids = []
        binnies = []
        for i, seq in enumerate(self.seq_indices):
            for pic in range(self.pics_per_seq[i]):
                images.append('./data/tripod_seq_{}_{}.jpg'.format(seq, _add_zeros(pic+1)))

            targets.append(self._generate_targets(i))
            one, two = self._discretized_labels(targets[i])
            labels.append(one)
            binnies.append(two)
            image_seq.append(images)
            images = []

        seq_ids_list = []
        for seq in range(self.num_seqs):
            seq_ids_list.append([seq + 1] * self.pics_per_seq[seq])   
        
        for seq_list in seq_ids_list:
            for seq_id in seq_list:
                seq_ids.append(seq_id)       
            
        for seq in image_seq:
            for img in seq:
                self.images.append(img)

        for seq in targets:
            for target in seq:
                self.targets.append(target)

        for seq in labels:
            for l in seq:
                self.labels.append(l)

        for b in binnies:
            for bb in b:
                self.binnies.append(bb)

        self.samples = [self.images, self.targets, self.labels, self.binnies, seq_ids]
        #self.samples = zip(*self.samples)

        with open('epfl_targets.csv', 'w') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerows(self.samples)

        #self.samples = np.asarray(self.samples)
        
    def _generate_targets(self, seq_i):
        """
        Finds the frontal (south) facing index, and subtracts and adds degrees from both ends to create the targets
        :param seq:
        :param seq_i:
        :return target for a given sequence:
        """
        
        frontal_facing_idx = self.seq_frontal_index[seq_i]
        
        # load times for this sequence
        info = open('data/times.txt', 'r')
        times = info.readlines()
        times = [int(x) for x in times[seq_i].split()]
        info.close()
        
        degree_per_sec =  360.0 / times[self.pics_per_circ[seq_i]] 
        
        degs = []
        
        for idx in range(self.pics_per_seq[seq_i]):
            dt = abs(times[frontal_facing_idx-1] - times[idx])
            if(idx < frontal_facing_idx-1):
                d = 180 - dt * degree_per_sec * self.seq_direction[seq_i]
                degs.append(d % 360)
            else:
                d = 180 + dt * degree_per_sec * self.seq_direction[seq_i]
                degs.append(d % 360)
                
            
        return degs

    def _discretized_labels(self, tgts):
        disc_targets = []
        within_Range = lambda x, hi, lo: True if hi > x >= lo else False
        temp_bins = [(360 / self.bins) * b % 360 for b in range(self.bins)]  # generates bin end angles
        temp_bins.append(360.0)  # adds 360 at the end to complete the circle
        degree_swath = 360 / self.bins / self.rotations  # how much the targets will be rotated by per rotation
        bin_angles = []
        for img in range(len(tgts)):
            temp_disc_targets = np.zeros((self.rotations, self.bins))  # each img gets a target array
            for rot in range(self.rotations):
                printable_bins = [(b+20*rot)%360 for b in temp_bins]
                bin_angles.append(printable_bins[:-1])
                tgt = (tgts[img] - (rot * degree_swath)) % 360  # rotate targets back based on which rotation we are in
                for i in range(len(temp_bins[:-1])):  # step through bins
                    if within_Range(tgt, temp_bins[i+1], temp_bins[i]):  # if within bin, mark it
                        temp_disc_targets[rot, i] = 1
            disc_targets.append(temp_disc_targets)
        bin_angles = np.asarray(bin_angles)
        #print(bin_angles.shape)
        bin_angles = bin_angles.reshape((-1,self.rotations*self.bins))
        #print(bin_angles.shape)
        return disc_targets, bin_angles

if __name__ == '__main__':
    data = Data(3,6)
    # for i in range(100):
    #     print(data.samples[2][i],'\n')