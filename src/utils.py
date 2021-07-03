
import torch
import matplotlib
import matplotlib.pyplot as plt


DCT_INTENT_CLASS = {
    'tour_offer': 0,
    'recommend_activity, recommend_poi': 1,
    'book_tour': 2,
    'recommend_activity': 3,
    'account_information': 4,
    'reservation_regulations': 5,
    'tour_inquiry_pick_up': 6,
    'tour_inquiry': 7,
    'describe_location, recommend_poi': 8,
    'reservation_inquiry': 9,
    'restaurant_inquiry': 10,
    'recommend_poi': 11,
    'restaurant_reservation_regulations': 12,
    'tour_available': 13,
    'tour_cancel': 14,
    'generic_request': 15,
    'reservation_update_request': 16,
    'needs_assistance': 17
    }

DCT_CLASS_INTENT = {v:k for k,v in DCT_INTENT_CLASS.items()}

def convert_intent_class(intent):
    return DCT_INTENT_CLASS[intent]

def convert_class_intent(class_):
    return DCT_CLASS_INTENT[class_]

def plot_train_val_loss(train_loss, val_loss, output="loss.png"):
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color="orange", label="train loss")
    plt.plot(val_loss, color="red", label="validataion loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(output)