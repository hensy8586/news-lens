import streamlit as st


def card_css():
    """ Global CSS for cards + inline detail block """

    st.markdown(
        """
        <style>
        /* Make the link wrapper behave like a normal block (if we use it later) */
        .news-card-link-wrapper,
        .news-card-link-wrapper:link,
        .news-card-link-wrapper:visited,
        .news-card-link-wrapper:hover,
        .news-card-link-wrapper:active {
            text-decoration: none !important;
            color: inherit !important;
            display: block;
        }

        /* --- Collapsible container --- */
        .news-details {
            margin-bottom: 1.25rem;
        }

        .news-details > summary {
            list-style: none;             /* remove default marker */
            cursor: pointer;
        }

        /* Hide the default triangle marker in most browsers */
        .news-details > summary::-webkit-details-marker {
            display: none;
        }

        /* Card base */
        .news-card {
            border-radius: 0.75rem;
            padding: 1rem 1.25rem;
            background-color: #111827;          /* dark card (good in dark mode) */
            border: 1px solid #1f2937;
            box-shadow: 0 4px 10px rgba(0,0,0,0.25);
            transition: background-color 0.15s ease,
                        transform 0.15s ease,
                        box-shadow 0.15s ease;
        }

        .news-details[open] .news-card {
            background-color: #1f2937;          /* slightly lighter when open */
            box-shadow: 0 8px 20px rgba(0,0,0,0.35);
            transform: translateY(-1px);
        }

        .news-card:hover {
            background-color: #1f2937;
            box-shadow: 0 8px 20px rgba(0,0,0,0.35);
            transform: translateY(-2px);
        }

        /* Vertical card layout: image on top, text below */
        .news-card-inner {
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }

        .news-card-image {
            width: 100%;
        }

        .news-card-image img {
            width: 100%;
            height: auto;         /* keep full image; no top/bottom cropping */
            max-height: 220px;    /* cap so cards don't get huge */
            border-radius: 0.5rem;
            display: block;
            object-fit: contain;  /* letterbox if needed, preserve full image */
        }

        @media (max-width: 768px) {
            .news-card-inner {
                flex-direction: column;
            }
            .news-card-image {
                width: 100%;
            }
            .news-card-image img {
                width: 100%;
                height: auto;
            }
        }

        .news-card-content {
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
        }

        .news-card-title {
            font-size: 1.05rem;
            font-weight: 600;
            margin-bottom: 0.1rem;
        }

        .news-card-meta {
            font-size: 0.8rem;
            color: #9ca3af;
        }

        /* --- Inline detail block below the card --- */
        .news-detail-panel {
            margin-top: 0.5rem;
            padding: 0.85rem 1rem;
            background-color: #020617;
            border-radius: 0.75rem;
            border: 1px solid #1f2937;
            box-shadow: 0 4px 12px rgba(0,0,0,0.35);
            position: relative;
            font-size: 0.9rem;
            color: #e5e7eb;
        }

        /* Arrow pointing up to the card */
        .news-detail-panel::before {
            content: "";
            position: absolute;
            top: -8px;
            left: 32px;
            border-width: 8px;
            border-style: solid;
            border-color: #020617 transparent transparent transparent;
        }

        .news-detail-summary {
            margin-bottom: 0.75rem;
            white-space: pre-wrap;
        }

        .news-detail-link a {
            color: #60a5fa;
            text-decoration: none;
            font-size: 0.85rem;
        }

        .news-detail-link a:hover {
            text-decoration: underline;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
