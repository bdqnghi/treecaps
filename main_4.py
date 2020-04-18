import os
import logging
import pickle
import tensorflow as tf
import numpy as np
import network as network
import sampling as sampling
import sys
import random
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from data_loader import MonoLanguageProgramData
import argparse
import random
import shutil
from keras_radam.training import RAdamOptimizer
from network_4 import TreeCapsModel
import config
random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--train_batch_size', type=int, default=5, help='train batch size, always 1')
parser.add_argument('--test_batch_size', type=int, default=1, help='test batch size, always 1')
parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE, help='test batch size, always 1')
parser.add_argument('--niter', type=int, default=config.EPOCHS, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--verbal', type=bool, default=True, help='print training info or not')
parser.add_argument('--n_classes', type=int, default=config.N_CLASSES, help='manual seed')
parser.add_argument('--train_directory', default=config.TRAIN_DIRECTORY, help='train program data') 
parser.add_argument('--test_directory', default=config.TEST_DIRECTORY, help='test program data')
parser.add_argument('--val_directory', default=config.VAL_DIRECTORY, help='val program data')
parser.add_argument('--validating', type=int, default=config.VALIDATING, help='validate when training or not')
parser.add_argument('--model_path', default="model", help='path to save the model')
parser.add_argument('--analysis_path', default="analysis", help='path to save the analysis files')
parser.add_argument('--training', action="store_true",help='is training')
parser.add_argument('--testing', action="store_true",help='is testing')
parser.add_argument('--analyzing', action="store_true",help='is analying')
parser.add_argument('--training_percentage', type=float, default=1.0 ,help='percentage of data use for training')
parser.add_argument('--log_path', default="" ,help='log path for tensorboard')
parser.add_argument('--embeddings_directory', default="embedding/node_type_lookup.pkl")
parser.add_argument('--cuda', default=config.CUDA,type=str, help='enables cuda')
parser.add_argument('--num_conv', type=int, default=config.NUM_CONV)
parser.add_argument('--node_type_dim', type=int, default=config.NODE_TYPE_DIM)
parser.add_argument('--output_size', type=int, default=config.OUTPUT_SIZE)
parser.add_argument('--code_caps_num_caps', type=int, default=config.CODE_CAPS_NUM_CAPS)
parser.add_argument('--code_caps_output_dimension', type=int, default=config.CODE_CAPS_OUTPUT_DIMENSION)
parser.add_argument('--class_caps_output_dimension', type=int, default=config.CLASS_CAPS_OUTPUT_DIMENSION)


opt = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = opt.cuda

if not os.path.isdir("cached"):
    os.mkdir("cached")

batch_size = opt.batch_size


def form_model_path(opt):
    model_traits = {}
    model_traits["output_size"] = config.OUTPUT_SIZE
    model_traits["num_conv"] = config.NUM_CONV
    model_traits["n_class"] = config.N_CLASSES
    model_traits["code_caps_num_caps"] = config.CODE_CAPS_NUM_CAPS
    model_traits["code_caps_o_dimension"] = config.CODE_CAPS_OUTPUT_DIMENSION
    model_traits["class_caps_o_dimension"] = config.CLASS_CAPS_OUTPUT_DIMENSION

    model_path = []
    for k, v in model_traits.items():
        model_path.append(k + "_" + str(v))
    
    # mnp = method_name_prediction to handle too long file name
    return "tree_caps_pc4" + "_" + "-".join(model_path)




def train_model(train_trees, val_trees, embedding_lookup, opt):
    random.shuffle(train_trees)
    random.shuffle(val_trees)

    
    # batch_size = opt.train_batch_size
    epochs = opt.niter

    treecaps = TreeCapsModel(opt)   

    optimizer = RAdamOptimizer(opt.lr)
    training_point = optimizer.minimize(treecaps.loss)
    
     ### init the graph
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    with tf.name_scope('saver'):
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(opt.model_path)
        if ckpt and ckpt.model_checkpoint_path:
            print("Continue training with old model")
            saver.restore(sess, ckpt.model_checkpoint_path)
            for i, var in enumerate(saver._var_list):
                print('Var {}: {}'.format(i, var))

    checkfile = os.path.join(opt.model_path, 'tree_network.ckpt')

    max_acc = 0.0
    model_accuracy_path = os.path.join(opt.model_path, "accuracy.txt")
    if not os.path.exists(model_accuracy_path):
        try:
            os.mkdir(opt.model_path)
        except Exception as e:
            print(e) 
        with open(model_accuracy_path, "w") as f:
            f.write("0.0")
    else:
        with open(model_accuracy_path, "r") as f1:
            data = f1.readlines()
            for line in data:
                max_acc = float(line.replace("\n",""))


    print("Begin training..........")
    
    cur_acc = 0.0
    for epoch in range(1, epochs+1):
        # bar = progressbar.ProgressBar(maxval=len(train_trees), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        # bar.start()
        for train_step, batch in enumerate(sampling.gen_samples(train_trees, opt.label_size, embedding_lookup, batch_size)):
            print("-------------")
            nodes, children, batch_labels = batch

           
            _, err = sess.run(
                [training_point, treecaps.loss],
                feed_dict={
                    treecaps.placeholders["node_types"]: nodes,
                    treecaps.placeholders["children_indices"]: children,
                    treecaps.placeholders["labels"]: batch_labels         
                }
            )
            # scores = sess.run(
            #     [treecaps.code_caps],
            #     feed_dict={
            #         treecaps.placeholders["node_types"]: nodes,
            #         treecaps.placeholders["children_indices"]: children,
            #         treecaps.placeholders["labels"]: batch_labels  
            #     }
            # )
            # print(scores[0].shape)
            # print(conv_output.shape)
            # print(attention_scores.shape)

            print("Epoch:", epoch, "Step:", train_step, "Loss:", err, "Current Acc: ", cur_acc, "Max Acc: ",max_acc)
      
            if opt.validating == 0:
                if train_step % config.CHECKPOINT_EVERY == 0 and train_step > 0:
                    saver.save(sess, checkfile)
                    print("Saved model....")

            if opt.validating == 1:
                if train_step % config.CHECKPOINT_EVERY == 0 and train_step > 0:
                    print("Validating.........")
                    correct_labels = []
                    predictions = []
                    logits = []
                    for batch in sampling.gen_samples(val_trees, opt.label_size, embedding_lookup, batch_size):
                        print("-----------")
                        nodes, children, batch_labels = batch


                        output = sess.run([treecaps.softmax],
                            feed_dict={
                                treecaps.placeholders["node_types"]: nodes,
                                treecaps.placeholders["children_indices"]: children,
                                treecaps.placeholders["labels"]: batch_labels  
                            }
                        )
                        batch_correct_labels = np.argmax(batch_labels, axis=1).tolist()
                        batch_predictions = np.argmax(output[0], axis=1).tolist()
                        correct_labels.extend(batch_correct_labels)
                        predictions.extend(batch_predictions)
                        print("Ground truth : " + str(batch_correct_labels))
                        print("Predictions : " + str(batch_predictions))

                    target_names = list(opt.labels)
                    acc = accuracy_score(correct_labels, predictions)
                    cur_acc = acc
                    
                    if (acc>max_acc):
                        max_acc = acc
                        saver.save(sess, checkfile)
                        print("Saved model....")

                        with open(model_accuracy_path,"w") as f2:
                            f2.write(str(max_acc))

                    print('Epoch',str(epoch),'Accuracy:', acc, 'Max Acc: ',max_acc)

    print("Finish all iters, storring the whole model..........")


def test_model(test_trees, embedding_lookup, opt):
    
   
    epochs = opt.niter

    random.shuffle(test_trees)

    # build the inputs and outputs of the network
    treecaps = TreeCapsModel(opt)   
 
 
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    with tf.name_scope('saver'):
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(opt.model_path)
        if ckpt and ckpt.model_checkpoint_path:
            print("Continue training with old model")
            saver.restore(sess, ckpt.model_checkpoint_path)
            for i, var in enumerate(saver._var_list):
                print('Var {}: {}'.format(i, var))
                
    checkfile = os.path.join(opt.model_path, 'tree_network.ckpt')

    correct_labels = []
    predictions = []
    print('Computing training accuracy...')
    for batch in sampling.gen_samples(test_trees, opt.label_size, embedding_lookup, batch_size):
        nodes, children, batch_labels = batch
        print("----------------------------------")
        output = sess.run([treecaps.softmax],
            feed_dict={
                treecaps.placeholders["node_types"]: nodes,
                treecaps.placeholders["children_indices"]: children,
                treecaps.placeholders["labels"]: batch_labels  
            }
        )
        # print(output[0].shape)
     
        batch_correct_labels = np.argmax(batch_labels, axis=1).tolist()
        batch_predictions = np.argmax(output[0], axis=1).tolist()
        correct_labels.extend(batch_correct_labels)
        predictions.extend(batch_predictions)
        print("Ground truth : " + str(batch_correct_labels))
        print("Predictions : " + str(batch_predictions))
        print('Accuracy:', accuracy_score(correct_labels, predictions))
      

    target_names = list(opt.labels)
    print(classification_report(correct_labels, predictions, target_names=target_names))
    print(confusion_matrix(correct_labels, predictions))
    print('*'*50)
    print('Accuracy:', accuracy_score(correct_labels, predictions))
    print('*'*50)


def analysis(test_trees, embedding_lookup, opt):
    
   
    epochs = opt.niter

    random.shuffle(test_trees)

    # build the inputs and outputs of the network
    treecaps = TreeCapsModel(opt)   
 
 
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    with tf.name_scope('saver'):
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(opt.model_path)
        if ckpt and ckpt.model_checkpoint_path:
            print("Continue training with old model")
            saver.restore(sess, ckpt.model_checkpoint_path)
            for i, var in enumerate(saver._var_list):
                print('Var {}: {}'.format(i, var))
                
    checkfile = os.path.join(opt.model_path, 'tree_network.ckpt')

    try:
        os.makedirs(opt.analysis_path, exist_ok=True)
    except Exception as e:
        print(e)

    code_caps_analysis_path = os.path.join(opt.analysis_path, "code_caps.txt")
    class_caps_analysis_path = os.path.join(opt.analysis_path, "class_caps.txt")

    correct_labels = []
    predictions = []
    print('Computing training accuracy...')
    
    for batch in sampling.gen_samples(test_trees, opt.label_size, embedding_lookup, batch_size):
        nodes, children, batch_labels = batch
        output = sess.run([treecaps.code_caps, treecaps.class_caps],
            feed_dict={
                treecaps.placeholders["node_types"]: nodes,
                treecaps.placeholders["children_indices"]: children,
                treecaps.placeholders["labels"]: batch_labels  
            }
        )

        code_caps_score = output[0]
        class_caps_score = output[1]
        # print(output[0].shape)
        
        batch_correct_labels = np.argmax(batch_labels, axis=1).tolist()
        correct_label = batch_correct_labels[0]
        code_caps_score = np.squeeze(code_caps_score, axis=0)
        code_caps_score = np.squeeze(code_caps_score, axis=2)


        class_caps_score = np.squeeze(class_caps_score, axis=0)
        class_caps_score = np.squeeze(class_caps_score, axis=2)


        print(code_caps_score.shape)

        
        for channel, code_capsule in enumerate(code_caps_score):
            line = str(correct_label) + "," + str(channel) + ","
            # capsule = capsule.tolist()
            capsule_score = []
            for score in code_capsule:
                capsule_score.append(str(score))

            capsule_score = " ".join(capsule_score)
            with open(code_caps_analysis_path, "a") as f:
                line = line + capsule_score
                f.write(line)
                f.write("\n")

        for label, class_capsule in enumerate(class_caps_score):

            line = str(label) + ","
            # capsule = capsule.tolist()
            capsule_score = []
            for score in class_capsule:
                capsule_score.append(str(score))

            capsule_score = " ".join(capsule_score)
            with open(class_caps_analysis_path, "a") as f:
                line = line + capsule_score
                f.write(line)
                f.write("\n")

# def test(trees , name):
#     all_tuples = []
#     for tree in trees:
#         tup = (tree["size"],tree["label"])
#         all_tuples.append(tup)
#     all_tuples = sorted(all_tuples, key=lambda tup: tup[0])
#     with open("tree_size_" + name + ".txt", "w") as f:
#         for t in all_tuples:
#             line = str(t[0]) + "," + str(t[1])
#             # print(tree["size"])
#             f.write(line)
#             f.write("\n")
def main(opt):
    
    print("Loading embeddings....")
    with open(opt.embeddings_directory, 'rb') as fh:
        node_type_lookup = pickle.load(fh,encoding='latin1')
    opt.node_type_lookup = node_type_lookup

    model_directory = form_model_path(opt)
    opt.model_directory = model_directory
    opt.model_path = os.path.join(opt.model_path, model_directory)
    opt.analysis_path = os.path.join(opt.analysis_path, model_directory)

    labels = [str(i) for i in range(0, opt.n_classes)]
    opt.label_size = len(labels)
    opt.labels = labels

    if os.path.exists(opt.model_path):
        print("Continue using old model : " + str(opt.model_path))

    if opt.training:
        print("Loading train trees...")
        train_data_loader = MonoLanguageProgramData(opt.train_directory, 0, opt.n_classes)
        train_trees = train_data_loader.trees

        # test(train_trees, "train")
        val_data_loader = MonoLanguageProgramData(opt.test_directory, 2, opt.n_classes)
        val_trees = val_data_loader.trees

        val_trees = random.sample(val_trees, int(len(val_trees)/8))

        train_model(train_trees, val_trees, node_type_lookup , opt) 

    if opt.testing:
        print("Loading test trees...")
        test_data_loader = MonoLanguageProgramData(opt.test_directory, 1, opt.n_classes)
        test_trees = test_data_loader.trees
        print("All testing trees : " + str(len(test_trees)))
        test_model(test_trees, node_type_lookup , opt) 

    if opt.analyzing:
        print("Loading test trees...")
        test_data_loader = MonoLanguageProgramData(opt.test_directory, 1, opt.n_classes)
        test_trees = test_data_loader.trees
        print("All testing trees : " + str(len(test_trees)))
        analysis(test_trees, node_type_lookup , opt) 

if __name__ == "__main__":
    if not os.path.exists(opt.model_path):
        os.makedirs(opt.model_path)
    csv_log = open(opt.model_path+'/log.csv', "w")
    csv_log.write('Epoch,Training Loss,Validation Accuracy\n')
    main(opt)
    csv_log.close()