"""
Sample from a trained model
"""

import streamlit as st
from contextlib import nullcontext
import torch
import tiktoken
from model import GPT, GPTConfig

from helpers import (
    create_token_visualization,
    create_embedding_visualization,
    visualize_attention_matrices,
    plot_gelu,
)

st.set_page_config(layout="wide")
# -----------------------------------------------------------------------------
start = "The universe number is "
max_new_tokens = 1
temperature = 0.3
top_k = None
seed = 1337
device = "cpu"  # "cuda"
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)
# -----------------------------------------------------------------------------


@st.cache_resource
def init_torch():
    print("init_torch")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


@st.cache_resource
def get_init_tokenizer():
    print("Initizing tokenizer")
    enc = tiktoken.get_encoding("gpt2")
    return enc


@st.cache_resource
def get_init_model():
    # init from a given GPT-2 model
    print("Initizing model")
    model = GPT.from_pretrained("gpt2", dict(dropout=0.0))
    model.eval()
    model.to(device)
    return model


def encode(enc, s):
    # print(f"Encoding: {s}")
    return enc.encode(s, allowed_special={"<|endoftext|>"})


def decode(enc, l):
    # print(f"Decoding: {l}")
    return enc.decode(l)


#######################
# CSS styling
st.markdown(
    """
<style>
body {
    background-color: #0E1117;
    color: #C6CDD4;
}
</style>
""",
    unsafe_allow_html=True,
)


#######################
# Main Panel
def generate_visualization():
    global model, tokenizer, device, ctx, max_new_tokens, temperature, top_k, seed, dtype
    # Ensure input_text is not empty
    if not st.session_state.input_text:
        st.error("Please enter a prompt.")
        return

    # Run the model
    input_ids = encode(tokenizer, st.session_state.input_text)

    # Condition if the input is too long to normally show.
    if len(input_ids) >= 32:
        st.error("Input text is too long. Please enter a shorter prompt.")
        return

    x = torch.tensor(input_ids, dtype=torch.long, device=device)[None, ...]
    with torch.no_grad():
        with ctx:
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            generated_text = decode(tokenizer, y[0].tolist())
            # print(f"generating string: '{generated_text}'")

    t = x.size(1)
    # 1. Tokenization
    st.markdown("## 1. Tokenization Process")
    st.latex(r"x = \text{tokenizer.encode(input\_text)}")
    st.latex(r"x \in \mathbb{R}^{(1, " + str(t) + r")}")
    create_token_visualization(st.session_state.input_text, input_ids, tokenizer)

    # 2. Embedding Transformation + Positional Encoding
    st.markdown("## 2. Embedding Transformation + Positional Encoding")
    with torch.no_grad():
        with ctx:
            device = x.device
            b, t = x.size()
            pos = torch.arange(0, t, dtype=torch.long, device=device)
            tok_emb = model.transformer.wte(x)
            pos_emb = model.transformer.wpe(pos)
            dim = tok_emb.size(-1)
            x = tok_emb + pos_emb

            st.latex(r"\text{tok\_emb} = W_{te} \cdot x")
            st.latex(
                r"\text{tok\_emb} \in \mathbb{R}^{(" + str(t) + r", " + str(dim) + r")}"
            )
            st.latex(r"\text{pos\_emb} = W_{pe} \cdot \text{pos}")
            st.latex(
                r"\text{pos\_emb} \in \mathbb{R}^{(" + str(t) + r", " + str(dim) + r")}"
            )
            st.latex(r"\text{combined\_emb} = \text{tok\_emb} + \text{pos\_emb}")
            st.latex(
                r"\text{combined\_emb} \in \mathbb{R}^{("
                + str(t)
                + r", "
                + str(dim)
                + r")}"
            )

            tokens = [tokenizer.decode([id]) for id in input_ids]
            create_embedding_visualization(
                tokens,
                tok_emb[0],  # Remove batch dimension
                pos_emb,
                tok_emb[0] + pos_emb,
                tokenizer,
            )
    # 3. Transformer Layers
    st.markdown("## 3. Transformer Layers")

    st.markdown("#### 3.1 Layer Normalization")
    st.latex(r"\text{LayerNorm}(x) = \text{LN}(x)")
    st.latex(r"\text{LN}(x) \in \mathbb{R}^{(" + str(t) + r", " + str(dim) + r")}")

    st.markdown("#### 3.2 Multi-Head Self-Attention")
    st.latex(
        r"Q, K, V \in \mathbb{R}^{("
        + str(t)
        + r" \times "
        + str(dim // model.transformer.h[0].attn.n_head)
        + r")}"
    )
    st.latex(
        r"\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V"
    )
    st.latex(
        r"\text{where } QK^T \in \mathbb{R}^{(" + str(t) + r" \times " + str(t) + r")}"
    )

    st.latex(
        r"\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O"
    )
    st.latex(
        r"\text{where } \text{head}_i \in \mathbb{R}^{("
        + str(t)
        + r" \times "
        + str(dim // model.transformer.h[0].attn.n_head)
        + r")}"
    )

    # Use session state to store and persist layer selection
    if "layer_number" not in st.session_state:
        st.session_state.layer_number = 1

    # Dropdown to select layer number
    st.session_state.layer_number = st.selectbox(
        "Select Transformer Layer",
        list(range(1, GPTConfig.n_layer + 1)),
        index=st.session_state.layer_number - 1,
        key=f"layer_select_{st.session_state.layer_number}",
    )

    show_layer = st.session_state.layer_number - 1
    # attention visualized for selected layer and all heads
    with torch.no_grad():
        with ctx:
            for block in model.transformer.h[: show_layer + 1]:
                tmp = block.ln_1(x)
                tmp, attn_M = block.attn(tmp, save_attn=True)

                x = x + tmp
                x = x + block.mlp(block.ln_2(x))

            # run all layers
            for block in model.transformer.h[show_layer + 1 :]:
                tmp = block.ln_1(x)
                tmp = block.attn(tmp)

                x = x + tmp
                x = x + block.mlp(block.ln_2(x))
            x = model.transformer.ln_f(x)
            logits = model.lm_head(x[:, [-1], :])
    attn_M = attn_M.squeeze(0)
    fig = visualize_attention_matrices(attn_M, tokens)
    st.plotly_chart(fig, use_container_width=False)

    st.markdown("#### 3.3 Layer Normalization")
    st.latex(r"\text{LayerNorm}(x) = \text{LN}(x)")
    st.latex(r"\text{LN}(x) \in \mathbb{R}^{(" + str(t) + r", " + str(dim) + r")}")

    st.markdown("#### 3.4 Multi-Layer Perceptron")

    # Get actual dimensions from model
    d = model.config.n_embd  # embedding dimension
    t = x.size(1)  # sequence length

    st.latex(r"x_{\text{input}} \in \mathbb{R}^{(" + str(t) + r", " + str(d) + r")}")
    st.latex(
        r"W_{\text{fc}} \in \mathbb{R}^{(" + str(d) + r", " + str(4 * d) + r")}, \quad "
        r"W_{\text{proj}} \in \mathbb{R}^{(" + str(4 * d) + r", " + str(d) + r")}"
    )
    st.latex(r"h = \text{GELU}(W_{\text{fc}} \cdot x + b_{\text{fc}})")
    st.latex(r"\text{MLP}(x) = W_{\text{proj}} \cdot h + b_{\text{proj}}")

    st.plotly_chart(plot_gelu(), use_container_width=False)

    return


init_torch()
model = get_init_model()
tokenizer = get_init_tokenizer()

# Sidebar configuration
st.sidebar.title("🤖 GPT-2 Token Generation")
st.session_state.input_text = st.sidebar.text_input(
    "Enter your prompt:", "The universe number is", key="prompt_input"
)
generate_button = st.sidebar.button("Generate", key="generate_button")

generate_visualization()
