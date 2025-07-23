from myutils import read_jsonl
import os
import matplotlib.pyplot as plt
def draw_loss(loss_path='output/best_model/history.json',is_save=True,is_show=False):
    history = read_jsonl(loss_path)
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('ResNet Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    if is_save:
        plt.savefig('output/best_model/loss.png')
        print(f'save loss to output/best_model/loss.png')
    if is_show:
        plt.show()
if __name__ == '__main__':
    # draw_loss('output/resnet_models/resnet_8ed1ef38800413ba50979f9dd7becc1a/history.json')
    draw_loss('output/resnet_models/resnet_f2c659a2ccf2f9674573fb67a0a52f56/history.json')