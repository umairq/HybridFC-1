from torch.utils.data import DataLoader, random_split
import numpy as np
class Data:
    def __init__(self, data_dir=None, subpath=None, prop=None, complete_data= False, emb_typ = "TransE", emb_file = "",
                 bpdp_dataset=False, full_hybrid = False):
        complete_dataset  = complete_data
        # Quick workaround as we happen to have duplicate triples.
        # None if load complete data, otherwise load parts of dataset with folders in wrong directory.
        emb_folder = ""
        if bpdp_dataset == True:
            emb_folder = "bpdp/"
            if full_hybrid:
                data_dir = "dataset/data/bpdp/data/copaal/"
                self.train_set = list((self.load_data(data_dir + "train/", data_type="train")))
                self.test_data = list((self.load_data(data_dir + "test/", data_type="test")))
            else:
                data_dir = "dataset/data/bpdp/"
                self.train_set = list((self.load_data(data_dir+"train/", data_type="train")))
                self.test_data = list((self.load_data(data_dir+"test/", data_type="test")))
        elif complete_dataset==True: # for the entire dataset
            self.train_set = list((self.load_data(data_dir+"complete_dataset/", data_type="train")))
            self.test_data = list((self.load_data(data_dir+"complete_dataset/", data_type="test")))
        elif prop != None: # for properties split based datasets
            self.train_set = list((self.load_data(data_dir + "properties_split/train/" + prop, data_type="train")))
            self.test_data = list((self.load_data(data_dir + "properties_split/test/" + prop, data_type="test")))
        # elif subpath == None:
        #     self.train_set = list((self.load_data(data_dir, data_type="train")))
        #     self.test_data = list((self.load_data(data_dir, data_type="test")))
        elif full_hybrid == True:
            self.train_set = list((self.load_data(data_dir + "train/" + subpath, data_type="train")))
            self.test_data = list((self.load_data(data_dir + "test/" + subpath, data_type="test")))
        else:
            self.train_set = list((self.load_data(data_dir+"data/train/"+subpath, data_type="train")))
            self.test_data = list((self.load_data(data_dir+"data/test/"+subpath, data_type="test")))

        # random split
        # test_size = len(self.test_data) - int(len(self.test_data) / 3)
        # valid_size = len(self.test_data) - (len(self.test_data) - int(len(self.test_data) / 3))
        # # adding validation set in the sets
        # self.test_data, self.valid_set = random_split(self.test_data, [test_size, valid_size])
        # adding validation set in the sets
        # self.test_data, self.valid_set  = self.test_data[0:test_size], self.test_data[test_size:len(self.test_data)+1]
        #generate test and validation sets
        self.test_data, self.valid_data = self.generate_test_valid_set(self, self.test_data)



        # factcheck predictions on train and test data
        if bpdp_dataset == True:
            self.train_set_pred = list(
                (self.load_data(data_dir+"train/", data_type="train_pred", pred=True)))
            self.test_data_pred = list(
                (self.load_data(data_dir+"test/", data_type="test_pred", pred=True)))

        elif complete_dataset==True:
            self.train_set_pred = list((self.load_data(data_dir+"complete_dataset/", data_type="train_pred", pred=True)))
            self.test_data_pred = list((self.load_data(data_dir+"complete_dataset/", data_type="test_pred", pred=True)))
            subpath = "complete_data"
        elif prop != None:
            self.train_set_pred = list((self.load_data(data_dir + "properties_split/train/" + prop, data_type="train_pred", pred=True)))
            self.test_data_pred = list((self.load_data(data_dir + "properties_split/test/" + prop, data_type="test_pred", pred=True)))
        elif subpath==None:
            self.train_set_pred = list((self.load_data(data_dir , data_type="train_pred",pred=True)))
            self.test_data_pred = list((self.load_data(data_dir , data_type="test_pred",pred=True)))
        elif full_hybrid == True:
            self.train_set_pred = list(
                (self.load_data(data_dir + "train/" + subpath, data_type="train_pred", pred=True)))
            self.test_data_pred = list(
                (self.load_data(data_dir + "test/" + subpath, data_type="test_pred", pred=True)))
        else:
            self.train_set_pred = list((self.load_data(data_dir+"data/train/"+subpath, data_type="train_pred",pred=True)))
            self.test_data_pred = list((self.load_data(data_dir+"data/test/"+subpath, data_type="test_pred",pred=True)))
        self.test_data_pred, self.valid_data_pred = self.generate_test_valid_set(self, self.test_data_pred)

        self.data = self.train_set + list(self.test_data)  + list(self.valid_data)
        self.entities = self.get_entities(self.data)
        # uncomment it later when needed
        if bpdp_dataset:
            if full_hybrid:
                if data_dir == 'dataset/data/copaal/':
                    data_dir = 'dataset/data/bpdp/data/copaal/'
                self.save_all_resources(self.entities, data_dir, "combined/",
                                        True)
            else:
                self.save_all_resources(self.entities, data_dir, "/combined/", True)
        elif prop != None:
            self.save_all_resources(self.entities, data_dir, "data/combined/properties_split/" + prop.replace("/","_"), True)
        elif full_hybrid == True:
            self.save_all_resources(self.entities, data_dir.replace("data/copaal", ""), "data/combined/" + subpath,
                                    True)
        else:
            self.save_all_resources(self.entities, data_dir, "data/combined/" + subpath, True)



        # self.relations = list(set(self.get_relations(self.train_set) + self.get_relations(self.test_data)))
        self.relations = self.get_relations(self.data)
        # uncomment it later  when needed
        if bpdp_dataset:
            self.save_all_resources(self.relations, data_dir, "/combined/", False)
            # exit(1)
        elif prop != None:
            self.save_all_resources(self.relations, data_dir, "data/combined/properties_split/" + prop.replace("/","_"), False)
        elif full_hybrid == True:
            self.save_all_resources(self.relations, data_dir.replace("data/copaal",""), "data/combined/" + subpath, False)
        else:
            self.save_all_resources(self.relations, data_dir, "data/combined/" + subpath, False)

        self.num_entities = len(self.entities)
        self.num_relations = len(self.relations)

        self.idx_entities = dict()
        self.idx_relations = dict()

        # Generate integer mapping
        for i in self.entities:
            self.idx_entities[i] = len(self.idx_entities)
        for i in self.relations:
            self.idx_relations[i] = len(self.idx_relations)


        self.emb_entities = self.get_embeddings(self.idx_entities,emb_file+'Embeddings/'+emb_typ+'/'+emb_folder,'all_entities_embeddings_final')
        self.emb_relation = self.get_embeddings(self.idx_relations,emb_file+'Embeddings/'+emb_typ+'/'+emb_folder,'all_relations_embeddings_final')
        if bpdp_dataset == True:
            self.emb_sentences_train1 = self.get_sent_embeddings(data_dir + "combined/", 'trainSE.csv',
                                                                 self.train_set)

            self.emb_sentences_train = self.update_sent_train_embeddings(self, self.emb_sentences_train1)
            self.emb_sentences_test, self.emb_sentences_valid = self.get_sent_test_valid_embeddings(
                data_dir + "combined/", 'testSE.csv', self.test_data, self.valid_data)
            self.emb_sentences_test = self.update_sent_train_embeddings(self, self.emb_sentences_test)
            self.emb_sentences_valid = self.update_sent_train_embeddings(self, self.emb_sentences_valid)
            if full_hybrid:
                self.copaal_veracity_score1 = self.get_copaal_veracity(data_dir + "combined/" , 'trainSE.csv',
                                                                       self.train_set)
                self.copaal_veracity_train = self.update_veracity_train_data(self, self.copaal_veracity_score1)
                self.copaal_veracity_test1, self.copaal_veracity_valid1 = self.get_veracity_test_valid_data(
                    data_dir + "combined/", 'testSE.csv', self.test_data, self.valid_data)
                self.copaal_veracity_test = self.update_veracity_train_data(self, self.copaal_veracity_test1)
                self.copaal_veracity_valid = self.update_veracity_train_data(self, self.copaal_veracity_valid1)

        elif complete_dataset == True:
            self.emb_sentences_train1 = self.get_sent_embeddings(data_dir + "complete_dataset/",'trainSE.csv', self.train_set)
            self.emb_sentences_train = self.update_sent_train_embeddings(self, self.emb_sentences_train1)
            self.emb_sentences_test, self.emb_sentences_valid = self.get_sent_test_valid_embeddings(
                data_dir + "complete_dataset/" , 'testSE.csv', self.test_data, self.valid_data)
            self.emb_sentences_test = self.update_sent_train_embeddings(self, self.emb_sentences_test)
            self.emb_sentences_valid = self.update_sent_train_embeddings(self, self.emb_sentences_valid)

        elif full_hybrid == True:
            print("to be updated")
            self.emb_sentences_train1 = self.get_sent_embeddings(data_dir + "train/" + subpath, 'trainSE.csv',
                                                                 self.train_set)
            self.copaal_veracity_score1 = self.get_copaal_veracity(data_dir + "train/" + subpath, 'trainSE.csv',
                                                                 self.train_set)
            self.emb_sentences_train = self.update_sent_train_embeddings(self, self.emb_sentences_train1)
            self.copaal_veracity_train = self.update_veracity_train_data(self,self.copaal_veracity_score1)

            self.emb_sentences_test1, self.emb_sentences_valid1 = self.get_sent_test_valid_embeddings(
                data_dir + "test/" + subpath, 'testSE.csv', self.test_data, self.valid_data)

            self.copaal_veracity_test1, self.copaal_veracity_valid1 = self.get_veracity_test_valid_data(
                data_dir + "test/" + subpath, 'testSE.csv', self.test_data, self.valid_data)
            self.emb_sentences_test = self.update_sent_train_embeddings(self, self.emb_sentences_test1)
            self.emb_sentences_valid = self.update_sent_train_embeddings(self, self.emb_sentences_valid1)

            self.copaal_veracity_test = self.update_veracity_train_data(self, self.copaal_veracity_test1)
            self.copaal_veracity_valid = self.update_veracity_train_data(self, self.copaal_veracity_valid1)

        elif prop == None:
            self.emb_sentences_train1 = self.get_sent_embeddings(data_dir+"data/train/"+subpath,'trainSE.csv', self.train_set)
            self.emb_sentences_train = self.update_sent_train_embeddings(self, self.emb_sentences_train1)
            self.emb_sentences_test, self.emb_sentences_valid = self.get_sent_test_valid_embeddings(
                data_dir + "data/test/" + subpath, 'testSE.csv', self.test_data, self.valid_data)
            self.emb_sentences_test = self.update_sent_train_embeddings(self, self.emb_sentences_test)
            self.emb_sentences_valid = self.update_sent_train_embeddings(self, self.emb_sentences_valid)

        else:
            self.emb_sentences_train1 = self.get_sent_embeddings(data_dir + "properties_split/train/" + prop, 'trainSE.csv', self.train_set)
            self.emb_sentences_train = self.update_sent_train_embeddings(self, self.emb_sentences_train1)
            self.emb_sentences_test, self.emb_sentences_valid = self.get_sent_test_valid_embeddings(
            data_dir + "properties_split/test/" + prop, 'testSE.csv', self.test_data, self.valid_data)
            self.emb_sentences_test = self.update_sent_train_embeddings(self, self.emb_sentences_test)
            self.emb_sentences_valid = self.update_sent_train_embeddings(self, self.emb_sentences_valid)

        self.idx_train_data = []
        i = 0
        for (s, p, o, label) in self.train_set:
            idx_s, idx_p, idx_o, label = self.idx_entities[s], self.idx_relations[p], self.idx_entities[o], label
            self.idx_train_data.append([idx_s, idx_p, idx_o, label , i])
            i = i + 1

        self.idx_valid_data = []
        j = 0
        for (s, p, o, label) in self.valid_data:
            idx_s, idx_p, idx_o, label = self.idx_entities[s], self.idx_relations[p], self.idx_entities[o], label
            self.idx_valid_data.append([idx_s, idx_p, idx_o, label,j])
            j = j + 1

        self.idx_test_data = []
        k = 0
        for (s, p, o, label) in self.test_data:
            idx_s, idx_p, idx_o, label = self.idx_entities[s], self.idx_relations[p], self.idx_entities[o], label
            self.idx_test_data.append([idx_s, idx_p, idx_o, label,k])
            k = k + 1

    def is_valid_test_available(self):
        if len(self.valid_data) > 0 and len(self.test_data) > 0:
            return True
        return False
    @staticmethod
    def save_all_resources(list_all_entities, data_dir, sub_path, entities):
        if entities:
            with open(data_dir+sub_path+'all_entities.txt',"w") as f:
                for item in list_all_entities:
                    f.write("%s\n" % item)
        else:
            with open(data_dir + sub_path + 'all_relations.txt', "w") as f:
                for item in list_all_entities:
                    f.write("%s\n" % item)

    @staticmethod
    def generate_test_valid_set(self, test_data):
        test_set = []
        valid_set = []
        i = 0
        sent_i = 0
        for data in test_data:
            if i % 20 == 0:
                valid_set.append(data)
            else:
                test_set.append(data)

            i += 1
        return  test_set, valid_set
    @staticmethod
    def load_data(data_dir, data_type, pred=False):
        try:
            data = []
            if pred == False:
                with open("%s%s.txt" % (data_dir, data_type), "r") as f:
                    for datapoint in f:
                        datapoint = datapoint.split()
                        if len(datapoint) == 4:
                            s, p, o, label = datapoint
                            if label == 'True':
                                label = 1
                            else:
                                label = 0
                            data.append((s, p, o, label))
                        elif len(datapoint) == 3:
                            s, p, label = datapoint
                            assert label == 'True' or label == 'False'
                            if label == 'True':
                                label = 1
                            else:
                                label = 0
                            data.append((s, p, 'DUMMY', label))
                        else:
                            raise ValueError
            else:
                with open("%s%s.txt" % (data_dir, data_type), "r") as f:
                    for datapoint in f:
                        datapoint = datapoint.split()
                        if len(datapoint) == 4:
                            s, p, o, label = datapoint
                            data.append((s, p, o, label))
                        elif len(datapoint) == 3:
                            s, p, label = datapoint
                            data.append((s, p, 'DUMMY', label))
                        else:
                            raise ValueError
        except FileNotFoundError as e:
            print(e)
            print('Add empty.')
            data = []
        return data

    @staticmethod
    def get_relations(data):
        relations = sorted(list(set([d[1] for d in data])))
        return relations

    @staticmethod
    def get_entities(data):
        entities = sorted(list(set([d[0] for d in data] + [d[2] for d in data])))
        return entities

    # / home / umair / Documents / pythonProjects / HybridFactChecking / Embeddings / ConEx_dbpedia
    @staticmethod
    def get_embeddings(idxs,path,name):
        embeddings = dict()
        # print("%s%s.txt" % (path,name))
        with open("%s%s.txt" % (path,name), "r") as f:
            for datapoint in f:
                data = datapoint.split('> ,')
                if len(data)==1:
                    data = datapoint.split('>\",')
                if len(data) > 1:
                    data2 = data[0]+">",data[1].split(',')
                    test = data2[0].replace("\"","").replace("_com",".com").replace("Will-i-am","Will.i.am").replace("Will_i_am","Will.i.am")
                    if test in idxs:
                        embeddings[test] = data2[1]
                    # else:
                    #     print('Not in embeddings:',datapoint)
                        # exit(1)
                # else:
                #     print('Not in embeddings:',datapoint)
                #     exit(1)
        for emb in idxs:
            if emb not in embeddings.keys():
                print("this is missing in embeddings file:"+ emb)
                exit(1)

        if len(idxs) > len(embeddings):
            print("embeddings missing")
            exit(1)
        embeddings_final = dict()
        for emb in idxs.keys():
            if emb in embeddings.keys():
                embeddings_final[emb] = embeddings[emb]
            else:
                print('no embedding', emb)
                exit(1)

        return embeddings_final.values()

    @staticmethod
    def get_copaal_veracity(path, name, train_data):
        emb = dict()

        embeddings_train = dict()
        # print("%s%s" % (path,name))

        i = 0
        train_i = 0
        found = False
        with open("%s%s" % (path, name), "r") as f:
            for datapoint in f:
                if datapoint.startswith("0,1,2"):
                    continue
                else:
                    emb[i] = datapoint.split(',')
                    try:
                        for dd in train_data:
                            # figure out some way to handle this first argument well
                            if (emb[i][0] == dd[0].replace(',', '')) and (emb[i][1] == dd[1].replace(',', '')) and (
                                    emb[i][2] == dd[2].replace(',', '')):
                                # print('train data found')
                                embeddings_train[train_i] =np.append(emb[i][:3],emb[i][-1].replace("\n",""))
                                train_i += 1
                                found = True
                                break

                            # else:
                            #     print('error')
                            # exit(1)
                    except:
                        print('ecception')
                        exit(1)
                    if found == False:
                        if (train_i >= len(train_data)):
                            break
                        else:
                            print("some training data missing....not found:" + str(emb[i]))
                            exit(1)
                    i = i + 1
                    found = False

                    # i = i+1
            embeddings_train_final = dict()
            jj = 0
            # print("sorting")
            for embb in train_data:
                ff = False
                for embb2 in embeddings_train.values():
                    if ((embb[0].replace(',', '') == embb2[0].replace(',', '')) and (
                            embb[1].replace(',', '') == embb2[1].replace(',', '')) and (
                            embb[2].replace(',', '') == embb2[2].replace(',', ''))):
                        embeddings_train_final[jj] = embb2
                        jj = jj + 1
                        ff = True
                        break
                if ff == False:
                    print("problem: not found")
                    exit(1)

        if len(train_data) != len(embeddings_train_final):
            print("problem")
            exit(1)
        return embeddings_train_final.values()

    @staticmethod
    def get_sent_embeddings(path, name, train_data):
        emb = dict()

        embeddings_train = dict()
        # print("%s%s" % (path,name))

        i = 0
        train_i = 0
        found = False
        with open("%s%s" % (path, name), "r") as f:
            for datapoint in f:
                if datapoint.startswith("0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20"):
                    continue
                else:
                    emb[i] = datapoint.split(',')
                    try:
                        for dd in train_data:
                            # figure out some way to handle this first argument well
                            if (emb[i][0] == dd[0].replace(',', '')) and (emb[i][1] == dd[1].replace(',', '')) and (
                                    emb[i][2] == dd[2].replace(',', '')):
                                # print('train data found')
                                if (len(emb[i])) == ((768 * 3) + 1):
                                    embeddings_train[train_i] = emb[i][:-1]
                                else:
                                    embeddings_train[train_i] = emb[i]
                                train_i += 1
                                found = True
                                break

                            # else:
                            #     print('error')
                            # exit(1)
                    except:
                        print('ecception')
                        exit(1)
                    if found == False:
                        if (train_i >= len(train_data)):
                            break
                        else:
                            print("some training data missing....not found:" + str(emb[i]))
                            exit(1)
                    i = i + 1
                    found = False

                    # i = i+1
            embeddings_train_final = dict()
            jj = 0
            # print("sorting")
            for embb in train_data:
                ff = False
                for embb2 in embeddings_train.values():
                    if ((embb[0].replace(',', '') == embb2[0].replace(',', '')) and (
                            embb[1].replace(',', '') == embb2[1].replace(',', '')) and (
                            embb[2].replace(',', '') == embb2[2].replace(',', ''))):
                        embeddings_train_final[jj] = embb2
                        jj = jj + 1
                        ff = True
                        break
                if ff == False:
                    print("problem: not found")
                    exit(1)

        if len(train_data) != len(embeddings_train_final):
            print("problem")
            exit(1)
        return embeddings_train_final.values()

    @staticmethod
    def update_copaal_veracity_score(self, train_emb):
        embeddings_train = dict()
        i = 0
        for train in train_emb:
            embeddings_train[i] = train[3:]
            i += 1

        return embeddings_train.values()

    @staticmethod
    def update_veracity_train_data(self, train_emb):
        embeddings_train = dict()
        i = 0
        for train in train_emb:
            embeddings_train[i] = train[3:]
            i += 1

        return embeddings_train.values()
    @staticmethod
    def update_sent_train_embeddings(self, train_emb):
        embeddings_train = dict()
        i=0
        for train in train_emb:
            embeddings_train[i] = train[3:]
            i+=1

        return embeddings_train.values()

    @staticmethod
    def get_veracity_test_valid_data(path, name, test_data, valid_data):
        embeddings_test, embeddings_valid = dict(), dict()
        emb = dict()
        # print("%s%s" % (path, name))
        found = False
        i = 0
        test_i = 0
        valid_i = 0
        with open("%s%s" % (path, name), "r") as f:
            for datapoint in f:
                if datapoint.startswith("0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20"):
                    continue
                else:
                    emb[i] = datapoint.split(',')
                    try:
                        for dd in test_data:
                            # figure out some way to handle this first argument well
                            if (emb[i][0].replace(',', '') == dd[0].replace(',', '')) and (
                                    emb[i][1].replace(',', '') == dd[1].replace(',', '')) and (
                                    emb[i][2].replace(',', '') == dd[2].replace(',', '')):
                                # print('test data found')
                                embeddings_test[test_i] = np.append(emb[i][:3],emb[i][-1].replace("\n",""))
                                test_i += 1
                                found = True
                                break
                        for vd in valid_data:
                            # figure out some way to handle this first argument well
                            if (emb[i][0].replace(',', '') == vd[0].replace(',', '')) and (
                                    emb[i][1].replace(',', '') == vd[1].replace(',', '')) and (
                                    emb[i][2].replace(',', '') == vd[2].replace(',', '')):
                                # print('valid data found')
                                embeddings_valid[valid_i] = np.append(emb[i][:3],emb[i][-1].replace("\n",""))
                                valid_i += 1
                                found = True
                                break
                        if found == False:
                            print("some data missing from test and validation sets..error" + str(emb[i]))
                            exit(1)
                        else:
                            found = False

                    except:
                        print('ecception')
                        exit(1)
                    i = i + 1

        embeddings_test_final, embeddings_valid_final = dict(), dict()
        i = 0
        for dd in test_data:
            for et in embeddings_test.values():
                if (et[0].replace(',', '') == dd[0].replace(',', '')) and (
                        et[1].replace(',', '') == dd[1].replace(',', '')) and (
                        et[2].replace(',', '') == dd[2].replace(',', '')):
                    embeddings_test_final[i] = et
                    i = i + 1
                    break
        i = 0
        for dd in valid_data:
            # print(dd)
            for et in embeddings_valid.values():
                if (et[0].replace(',', '') == dd[0].replace(',', '')) and (
                        et[1].replace(',', '') == dd[1].replace(',', '')) and (
                        et[2].replace(',', '') == dd[2].replace(',', '')):
                    embeddings_valid_final[i] = et
                    i = i + 1
                    break
        if (len(embeddings_valid_final) != len(valid_data)) and (len(embeddings_test_final) != len(test_data)):
            exit(1)
        return embeddings_test_final.values(), embeddings_valid_final.values()


    @staticmethod
    def get_sent_test_valid_embeddings(path, name, test_data, valid_data):
        embeddings_test, embeddings_valid = dict(),dict()
        emb = dict()
        # print("%s%s" % (path, name))
        found = False
        i = 0
        test_i = 0
        valid_i = 0
        with open("%s%s" % (path, name), "r") as f:
            for datapoint in f:
                if datapoint.startswith("0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20"):
                    continue
                else:
                    emb[i] = datapoint.split(',')
                    try:
                        for dd in test_data:
                        # figure out some way to handle this first argument well
                            if  (emb[i][0].replace(',', '') == dd[0].replace(',','')) and (emb[i][1].replace(',', '') == dd[1].replace(',','')) and (
                                    emb[i][2].replace(',', '') == dd[2].replace(',','')) :
                                # print('test data found')
                                embeddings_test[test_i] = emb[i]
                                test_i += 1
                                found = True
                                break
                        for vd in valid_data:
                            # figure out some way to handle this first argument well
                            if (emb[i][0].replace(',', '') == vd[0].replace(',', '')) and (
                                    emb[i][1].replace(',', '') == vd[1].replace(',', '')) and (
                                    emb[i][2].replace(',', '') == vd[2].replace(',', '')):
                                # print('valid data found')
                                embeddings_valid[valid_i] = emb[i]
                                valid_i += 1
                                found = True
                                break
                        if found == False:
                            print("some data missing from test and validation sets..error"+ str(emb[i]))
                            exit(1)
                        else:
                            found = False

                    except:
                        print('ecception')
                        exit(1)
                    i = i + 1

        embeddings_test_final, embeddings_valid_final = dict(), dict()
        i = 0
        for dd in test_data:
            for et in embeddings_test.values():
                if (et[0].replace(',', '') == dd[0].replace(',', '')) and (et[1].replace(',', '') == dd[1].replace(',', '')) and (
                        et[2].replace(',', '') == dd[2].replace(',', '')):
                    embeddings_test_final[i] = et
                    i = i + 1
                    break
        i = 0
        for dd in valid_data:
            # print(dd)
            for et in embeddings_valid.values():
                if (et[0].replace(',', '') == dd[0].replace(',', '')) and (et[1].replace(',', '') == dd[1].replace(',', '')) and (
                        et[2].replace(',', '') == dd[2].replace(',', '')):
                    embeddings_valid_final[i] = et
                    i = i + 1
                    break
        if (len(embeddings_valid_final)!= len(valid_data)) and (len(embeddings_test_final)!= len(test_data)):
            exit(1)
        return embeddings_test_final.values(), embeddings_valid_final.values()


        # return embeddings.values()



# # Test data class
# bpdp = True
# if not bpdp:
#     properties_split = ["deathPlace/","birthPlace/","author/","award/","foundationPlace/","spouse/","starring/","subsidiary/"]
#     datasets_class = ["range/","domain/","mix/","property/","domainrange/","random/"]
#     # make it true or false
#     prop_split = True
#     clss = datasets_class
#     if prop_split:
#         clss = properties_split
#
#     for cls in clss:
#         method = "emb-only" #emb-only  hybrid
#         path_dataset_folder = 'dataset/'
#         if prop_split:
#             dataset = Data(data_dir=path_dataset_folder, subpath= None, prop = cls)
#         else:
#             dataset = Data(data_dir=path_dataset_folder, subpath= cls)
# else:
#     path_dataset_folder = 'dataset/data/bpdp/'
#     dataset = Data(data_dir=path_dataset_folder, bpdp_dataset=True)
#     print("success")
