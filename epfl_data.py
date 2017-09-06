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
        
        info = open('tripod-seq/tripod-seq.txt', 'r')
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
        for i, seq in enumerate(self.seq_indices):
            for pic in range(self.pics_per_seq[i]):
                images.append('./tripod-seq/tripod_seq_{}_{}.jpg'.format(seq, _add_zeros(pic+1)))

            targets.append(self._generate_targets(i))
            labels.append(self._discretized_labels(targets[i]))
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
                


        self.samples = [self.images, self.targets, self.labels, seq_ids]
        self.samples = zip(*self.samples)

        with open('epfl_targets.csv', 'w') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerows(self.samples)

        self.samples = np.asarray(self.samples)
        
    def _generate_targets(self, seq_i):
        """
        Finds the frontal (south) facing index, and subtracts and adds degrees from both ends to create the targets
        :param seq:
        :param seq_i:
        :return target for a given sequence:
        """
        
        frontal_facing_idx = self.seq_frontal_index[seq_i]
        
        # load times for this sequence
        info = open('tripod-seq/times.txt', 'r')
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

        for img in range(len(tgts)):
            # print("IMG #:{} has TGT:{}".format(img, tgts[img]))
            temp_disc_targets = np.zeros((self.rotations, self.bins))
            temp_bins = []
            for rot in range(self.rotations):
                temp = [(360 / self.bins * (b + rot / 2) % 360) for b in range(self.bins)]
                temp.append(*[(360 + temp[-1] + 360 / self.bins) % 360 if (360 + temp[-1] + 360 / self.bins) % 360 != 0 else 360])
                temp_bins.append(temp)
                # print("TEMP BINS:", temp_bins, "\nROT:{}\n\n".format(rot))
                # print("CURRENT DEBUG:", temp_bins[rot])
                # print("CURRENT DEBUG2:", temp_bins[rot][0])
                for i in range(len((temp_bins[rot][:-1]))):
                    # print("BIN:",temp_bins[rot][i])
                    # print("HI:{} LO:{} TGT:{}".format(temp_bins[rot][i], temp_bins[rot][i+1], tgts[img]))
                    if within_Range(tgts[img], temp_bins[rot][i+1], temp_bins[rot][i]):
                        # print("HIT")
                        temp_disc_targets[rot,i] = 1
                # print("TARGETS:", temp_disc_targets)
            disc_targets.append(temp_disc_targets)
        return disc_targets

if __name__ == '__main__':
    data = Data(2,4)