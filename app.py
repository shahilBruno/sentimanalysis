import streamlit as st 
import torch
import torch.nn as nn
from transformers import AutoTokenizer


# ============= the model
import torch.nn as nn

class YelpRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(YelpRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids):
        embedded = self.dropout(self.embedding(input_ids))
        output, (hidden, cell) = self.lstm(embedded)
        cat_hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        return self.fc(cat_hidden)

@st.cache_resource
def load_resources():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = YelpRNN(tokenizer.vocab_size, embed_dim=128, hidden_dim=256, output_dim=5)
    model.load_state_dict(torch.load('model03.pth', map_location=device))
    model.to(device)
    model.eval() # Set to evaluation mode
    return tokenizer, model, device


# =========== web Ui ====================
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="⭐"
)
st.title("🍽️ Review Star Predictor")
st.markdown("Type a restaurant review below, and the RNN will predict the star rating.")

tokenizer, model, device = load_resources()

user_input = st.text_area("Enter your review:", placeholder="The food was amazing, but the service was slow...")

if st.button("Predict Rating"):
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        # Preprocessing
        inputs = tokenizer(
            user_input,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = inputs['input_ids'].to(device)

        # Inference
        with torch.no_grad():
            logits = model(input_ids)
            prediction = torch.argmax(logits, dim=1).item()
            probs = torch.nn.functional.softmax(logits, dim=1)

        # Output (Adjustment: 0-4 index back to 1-5 stars)
        stars = prediction + 1
        emoji = '⭐'
        st.success(f"### Predicted Rating: {stars*emoji}")
        
        # Optional: Show confidence levels
        st.write("Confidence per category:")
        cols = st.columns(5)
        for i in range(5):
            cols[i].metric(label=f"{i+1} ⭐", value=f"{probs[0][i]*100:.1f}%")

st.divider()
st.info(f"Running on: **{device.type.upper()}**")