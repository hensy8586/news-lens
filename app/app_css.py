import streamlit as st


def card_css():
    """ Global CSS for "card-like" style """

    st.markdown(
        """
        <style>
        /* Make the card wrapper behave like a normal block, not a blue/underlined link */
        .news-card-link-wrapper,
        .news-card-link-wrapper:link,
        .news-card-link-wrapper:visited,
        .news-card-link-wrapper:hover,
        .news-card-link-wrapper:active {
            text-decoration: none !important;
            color: inherit !important;
            display: block;
        }

        .news-card {
            border-radius: 0.75rem;
            padding: 1rem 1.25rem;
            margin-bottom: 1rem;
            background-color: #111827;          /* dark card (good in dark mode) */
            border: 1px solid #1f2937;
            box-shadow: 0 4px 10px rgba(0,0,0,0.25);
            transition: background-color 0.15s ease,
                        transform 0.15s ease,
                        box-shadow 0.15s ease;
        }

        .news-card:hover {
            background-color: #1f2937;          /* slightly lighter on hover */
            box-shadow: 0 8px 20px rgba(0,0,0,0.35);
            transform: translateY(-2px);
            cursor: pointer;
        }

        .news-card-inner {
            display: flex;
            gap: 1rem;
            align-items: flex-start;
        }

        .news-card-image {
            flex: 0 0 280px;
        }

        .news-card-image img {
            width: 100%;
            height: auto;
            border-radius: 0.5rem;
            display: block;
        }

        @media (max-width: 768px) {
            .news-card-inner {
                flex-direction: column;
            }
            .news-card-image {
                flex: 0 0 auto;
            }
            .news-card-image img {
                width: 100%;
                height: auto;
            }
        }

        .news-card-title {
            font-size: 1.05rem;
            font-weight: 600;
            margin-bottom: 0.25rem;
        }

        .news-card-meta {
            font-size: 0.8rem;
            color: #9ca3af;
            margin-bottom: 0.5rem;
        }

        .news-card-summary {
            font-size: 0.9rem;
            color: #e5e7eb;
            margin-bottom: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
