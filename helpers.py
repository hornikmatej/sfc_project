import streamlit as st
import torch
import plotly.graph_objs as go
import plotly.subplots as sp
import numpy as np
import torch.nn.functional as F


def plot_gelu():
    x = np.linspace(-5, 5, 100)
    y = F.gelu(torch.tensor(x, dtype=torch.float32)).numpy()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="GELU"))
    fig.update_layout(
        title="GELU Activation Function",
        xaxis_title="Input (x)",
        yaxis_title="GELU(x)",
        template="plotly_dark",
        width=600,
        height=400,
    )
    return fig


def create_token_visualization(input_text, input_ids, tokenizer):
    # CSS for the token boxes, arrows, and animations
    st.markdown(
        """
    <style>
        .container {
            display: flex;
            flex-direction: column;
            align-items: flex-start;
            margin: 20px 0;
            max-width: 800px;
        }
        .tokens-ids {
            display: grid;
            grid-template-columns: 200px 60px 100px;
            align-items: center;
            margin-bottom: 10px;
            width: 100%;
        }
        .token {
            justify-self: end;
            margin: 0;
            padding: 8px 12px;
            border: 1px solid #ccc;
            border-radius: 8px;
            background-color: #f9f9f9;
            box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
            animation: fadeIn 1s ease-in-out;
        }
        .id {
            justify-self: start;
            margin: 0;
            padding: 8px 12px;
            border: 1px solid #ccc;
            border-radius: 8px;
            background-color: #f9f9f9;
            box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
            animation: fadeIn 1s ease-in-out;
        }
        .arrow {
            width: 100%;
            height: 2px;
            background-color: #0078d7;
            position: relative;
            animation: fadeIn 1s ease-in-out;
        }
        .arrow:after {
            content: "";
            position: absolute;
            top: -4px;
            right: -1px;
            border-width: 4px;
            border-style: solid;
            border-color: transparent transparent transparent #0078d7;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
    """,
        unsafe_allow_html=True,
    )
    # Original Input Text
    st.markdown("📝 **Original Text:**")
    st.write(f'"{input_text}"')

    # Tokens and IDs with arrows
    st.markdown("#### 🔤 **Tokens and IDs Mapping**")
    tokens = [tokenizer.decode([id]) for id in input_ids]

    # Start container
    html_content = ['<div class="container">']
    for token, token_id in zip(tokens, input_ids):
        # Add each token-ID pair with an arrow
        html_content.append(
            f'<div class="tokens-ids">'
            f'<div class="token" style="color: black;">{token}</div>'
            f'<div class="arrow"></div>'
            f'<div class="id" style="color: black;">{token_id}</div>'
            f"</div>"
        )
    html_content.append("</div>")

    # Render the HTML
    st.markdown("".join(html_content), unsafe_allow_html=True)


def create_embedding_visualization(
    tokens, token_embeddings, pos_embeddings, combined_embeddings, tokenizer
):
    st.markdown(
        """
    <style>
        .embedding-container {
            margin: 20px 0;
        }
        .embedding-row {
            display: flex;
            align-items: center;
            margin: 10px 0;
            font-family: monospace;
        }
        .embedding-label {
            width: 150px;
            margin-right: 10px;
        }
        .embedding-values {
            display: flex;
            align-items: center;
            background-color: #f9f9f9;
            padding: 8px;
            border-radius: 4px;
            color: black;
        }
        .value {
            margin: 0 4px;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )

    def format_embedding(embedding):
        # Show first 5 values and dots
        values = [f"{v:.3f}" for v in embedding[:5].tolist()]
        return f"[{', '.join(values)}, ...]"

    st.markdown("### 📊 Token Embeddings")
    for i, (token, emb) in enumerate(zip(tokens, token_embeddings)):
        st.markdown(
            f"""
            <div class="embedding-container">
                <div class="embedding-row">
                    <div class="embedding-label">Token '{token}':</div>
                    <div class="embedding-values">{format_embedding(emb)}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 📍 Positional Embeddings")
    for i, emb in enumerate(pos_embeddings):
        st.markdown(
            f"""
            <div class="embedding-container">
                <div class="embedding-row">
                    <div class="embedding-label">Position {i}:</div>
                    <div class="embedding-values">{format_embedding(emb)}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 🔄 Combined Embeddings (Token + Position)")
    for i, (token, emb) in enumerate(zip(tokens, combined_embeddings)):
        st.markdown(
            f"""
            <div class="embedding-container">
                <div class="embedding-row">
                    <div class="embedding-label">Token '{token}' at {i}:</div>
                    <div class="embedding-values">{format_embedding(emb)}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def visualize_attention_matrices(attention_matrices, tokens=None, max_tokens=32):
    """
    Visualize attention matrices in a 4x3 grid using Plotly with top-left triangle.

    Parameters:
    - attention_matrices: torch.Tensor of shape (num_heads, seq_len, seq_len)
    - tokens: Optional list of tokens corresponding to the sequence
    - max_tokens: Maximum number of tokens to display (default 32)
    """
    # Ensure we're working with a numpy array
    if isinstance(attention_matrices, torch.Tensor):
        attention_matrices = attention_matrices.detach().cpu().numpy()

    # Truncate to max_tokens if necessary
    seq_len = attention_matrices.shape[1]
    effective_tokens = min(seq_len, max_tokens)

    # Generate default tokens if not provided
    if tokens is None:
        tokens = [f"Token {i}" for i in range(effective_tokens)]
    else:
        # Truncate or pad tokens to match sequence length
        tokens = tokens[:effective_tokens]
        if len(tokens) < effective_tokens:
            tokens += [f"Token {i}" for i in range(len(tokens), effective_tokens)]

    # Slice attention matrices to match effective tokens
    attention_matrices = attention_matrices[:, :effective_tokens, :effective_tokens]

    # Create mask for top-left triangle
    mask = np.triu(np.ones_like(attention_matrices[0], dtype=bool), k=1)

    # Dynamic scaling calculations
    base_height = 2000
    base_width = 2400
    base_text_size = 40
    base_max_tokens = max_tokens

    # Scale height, width, and text size based on sequence length
    scale_factor = min(1, effective_tokens / base_max_tokens)
    text_size = max(10, int(base_text_size * scale_factor))

    # Create subplot with dynamic spacing
    fig = sp.make_subplots(
        rows=4,
        cols=3,
        subplot_titles=[f"Head {i+1}" for i in range(12)],
        vertical_spacing=0.1,
        horizontal_spacing=0.02,
    )

    # Color scale
    colorscale = "Plasma"

    # Add heatmaps for each head
    for head in range(12):
        row = head // 3 + 1
        col = head % 3 + 1

        # Mask out the lower triangle
        masked_attention = np.ma.masked_array(attention_matrices[head], mask=mask)

        # Prepare weight text
        weight_text = [
            [
                f"{masked_attention[i,j]:.2f}" if not mask[i, j] else ""
                for j in range(effective_tokens)
            ]
            for i in range(effective_tokens)
        ]
        masked_attention = np.where(mask, None, masked_attention)

        heatmap = go.Heatmap(
            z=masked_attention,
            colorscale=colorscale,
            zmin=0,
            zmax=np.max(attention_matrices),
            text=weight_text,
            texttemplate="%{text}",
            textfont={"size": text_size},
            hoverinfo="text",
        )

        fig.add_trace(heatmap, row=row, col=col)

        # Set x tick labels for all heatmaps
        fig.update_xaxes(
            tickmode="array",
            tickvals=list(range(effective_tokens)),
            ticktext=tokens,
            tickangle=45,
            row=row,
            col=col,
        )

        # Set y tick labels only for the first column
        if col == 1:
            fig.update_yaxes(
                tickmode="array",
                tickvals=list(range(effective_tokens)),
                ticktext=tokens,
                row=row,
                col=col,
            )
        else:
            # Disable y-axis labels for other columns
            fig.update_yaxes(
                showticklabels=False,
                row=row,
                col=col,
            )

    # Update layout with dynamic dimensions
    fig.update_layout(
        title="Attention Matrices for Transformer",
        height=base_height,
        width=base_width,
        margin=dict(r=100, b=100),
    )

    return fig
