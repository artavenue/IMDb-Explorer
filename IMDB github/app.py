import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
from scipy import stats
import numpy as np

# --- Configuration ---
st.set_page_config(page_title="IMDb Explorer", layout="wide", page_icon="🎬")
DATA_DIR = "data"

# --- Helper Functions ---

@st.cache_data
def load_main_data():
    """
    Loads the optimized Movies & Series dataset.
    File: imdb_lite.csv.gz
    """
    file_path = os.path.join(DATA_DIR, "imdb_lite.csv.gz")
    
    if not os.path.exists(file_path):
        st.error(f"Data file not found: {file_path}. Please run 'create_lite_data.py' first.")
        return pd.DataFrame()
    
    # Load compressed CSV
    # Columns expected: tconst, titleType, primaryTitle, startYear, genres, averageRating, numVotes
    df = pd.read_csv(
        file_path, 
        compression='gzip',
        dtype={
            'startYear': 'Int32',
            'averageRating': 'float32',
            'numVotes': 'Int32'
        }
    )
    return df

@st.cache_data
def load_episode_data(show_tconst):
    """
    Loads episode ratings for a specific show from the filtered file.
    File: imdb_episodes_all.csv
    """
    file_path = os.path.join(DATA_DIR, "imdb_episodes_all.csv")
    
    if not os.path.exists(file_path):
        st.error(f"Episode file not found: {file_path}.")
        return pd.DataFrame()

    # We load the full episode file because it is now small.
    # This is faster than seeking through a CSV for every click.
    df = pd.read_csv(
        file_path,
        dtype={
            'seasonNumber': 'Int32', 
            'episodeNumber': 'Int32',
            'averageRating': 'float32',
            'numVotes': 'Int32'
        },
        na_values=["\\N"]
    )
    
    # Filter for the specific show
    return df[df['parentTconst'] == show_tconst].copy()

@st.cache_data
def prepare_genre_timeline_data(df_movies, min_year, max_year):
    """
    Prepares data for the genre timeline visualization.
    """
    # Filter by year range - ensure proper type conversion
    df_filtered = df_movies.copy()
    
    # Convert to regular int to avoid type issues
    df_filtered['startYear'] = pd.to_numeric(df_filtered['startYear'], errors='coerce').astype('Int64')
    
    # Apply year filter
    df_filtered = df_filtered[
        (df_filtered['startYear'] >= min_year) & 
        (df_filtered['startYear'] <= max_year)
    ]
    
    # Drop rows with missing genres or year
    df_filtered = df_filtered.dropna(subset=['genres', 'startYear'])
    
    # Explode the genres column to create one row per genre
    df_genre = df_filtered.copy()
    df_genre['genres'] = df_genre['genres'].str.split(',')
    df_genre = df_genre.explode('genres')
    
    # Group by year and genre, count movies
    genre_counts = df_genre.groupby(['startYear', 'genres']).size().reset_index(name='count')
    
    # Calculate percentage of movies per genre per year
    total_per_year = df_genre.groupby('startYear').size().reset_index(name='total')
    genre_timeline = pd.merge(genre_counts, total_per_year, on='startYear')
    genre_timeline['percentage'] = (genre_timeline['count'] / genre_timeline['total']) * 100
    
    return genre_timeline

@st.cache_data
def get_series_comparison_data(df_series):
    """
    Prepares data for series comparison analysis.
    """
    # Calculate series statistics
    series_stats = df_series.copy()
    
    # Add decade column
    series_stats['decade'] = (series_stats['startYear'] // 10) * 10
    
    # Explode genres for genre analysis
    series_genres = series_stats.copy()
    series_genres['genres'] = series_genres['genres'].str.split(',')
    series_genres = series_genres.explode('genres')
    
    return series_stats, series_genres

# --- Session State Initialization ---
if 'selected_show_tconst' not in st.session_state:
    st.session_state.selected_show_tconst = None
if 'navigation_mode' not in st.session_state:
    st.session_state.navigation_mode = "Movie Explorer"

# Check if coming from a URL query parameter (deep link to a series)
params = st.query_params
if "tconst" in params:
    st.session_state.selected_show_tconst = params["tconst"]
    st.session_state.navigation_mode = "TV Series Analyzer"

# --- Sidebar ---
st.sidebar.title("🎬 IMDb Explorer")
st.sidebar.markdown("""
**Features:**
* 🍿 **Movie Explorer**: Find hidden gems.
* 📺 **TV Series Analyzer**: Heatmaps & Trends.

<small>Created by Robert Mischuda</small>
""", unsafe_allow_html=True)
# Sidebar Navigation
mode = st.sidebar.radio(
    "Go to", 
    ["Movie Explorer", "TV Series Analyzer"],
    key="navigation_mode"
)

# --- Load Data ---
with st.spinner("Loading optimized data..."):
    df_main = load_main_data()

# Strictly Separate Movies and TV
# Note: 'tvMovie' is grouped with Movies. 'miniSeries' grouped with Series.
if not df_main.empty:
    df_movies = df_main[df_main['titleType'].isin(['movie', 'tvMovie'])].copy()
    df_series = df_main[df_main['titleType'].isin(['tvSeries', 'tvMiniSeries', 'miniSeries'])].copy()
else:
    df_movies = pd.DataFrame()
    df_series = pd.DataFrame()

# ==========================================
# PAGE 1: MOVIE EXPLORER
# ==========================================
if mode == "Movie Explorer":
    st.header("🍿 Movie Explorer")
    
    if df_movies.empty:
        st.warning("No movie data found. Check if create_lite_data.py ran correctly.")
    else:
        # --- Filters ---
        col1, col2 = st.columns(2)
        with col1:
            # Genre Filter
            all_genres = set()
            for g in df_movies['genres'].dropna():
                all_genres.update(g.split(','))
            selected_genres = st.multiselect("Filter by Genre", sorted(list(all_genres)))
        
        with col2:
            # Year Filter
            min_year = int(df_movies['startYear'].min()) if df_movies['startYear'].notna().any() else 1900
            max_year = int(df_movies['startYear'].max()) if df_movies['startYear'].notna().any() else 2025
            year_range = st.slider("Filter by Year", min_year, max_year, (min_year, max_year))

        # Apply Filters
        df_m_filtered = df_movies.copy()
        if selected_genres:
            mask = df_m_filtered['genres'].apply(lambda x: any(g in str(x).split(',') for g in selected_genres))
            df_m_filtered = df_m_filtered[mask]
        
        df_m_filtered = df_m_filtered[
            (df_m_filtered['startYear'] >= year_range[0]) & 
            (df_m_filtered['startYear'] <= year_range[1])
        ]
        
        # Sort by rating in descending order (highest first)
        df_m_filtered = df_m_filtered.sort_values(by='averageRating', ascending=False)
        
        # Search Bar
        search = st.text_input("Search Title...", "")
        if search:
            df_m_filtered = df_m_filtered[df_m_filtered['primaryTitle'].str.contains(search, case=False, na=False)]

        # Info Box
        st.info(f"Showing {len(df_m_filtered)} movies")

        # --- Tabs ---
        tab1, tab2, tab3, tab4 = st.tabs(["📄 Data Table", "💎 Hidden Gems", "📈 Golden Age", "🕒 Genre Timeline"])

        with tab1:
            # Prepare clean dataframe for display (Reset Index ensures sorting works)
            df_display = df_m_filtered[['primaryTitle', 'startYear', 'averageRating', 'numVotes', 'genres']].copy()
            df_display = df_display.reset_index(drop=True)
            
            st.dataframe(
                df_display,
                use_container_width=True,
                height=500,
                hide_index=True,
                column_config={
                    "primaryTitle": "Title",
                    "startYear": st.column_config.NumberColumn("Year", format="%d"),
                    "averageRating": st.column_config.NumberColumn("Rating", format="%.1f"),
                    "numVotes": "Votes",
                    "genres": "Genres"
                }
            )

        with tab2:
            st.markdown("### 💎 Find Hidden Gems (High Rating, Low Votes)")
            
            # Add helpful instructions
            st.markdown("""
            🎯 **What are Hidden Gems?**
            
            Hidden gems are fantastic movies that have flown under radar! They're highly-rated films that haven't received as many votes as blockbusters.
            
            📦 Draw a selection box around points in upper-left area (high rating, low votes). Your selected movies will appear in the table below.
        
            """)
            
            use_log_x = st.checkbox("Use Log Scale for Votes", value=True)
            
            if not df_m_filtered.empty:
                fig = px.scatter(
                    df_m_filtered,
                    x='numVotes',
                    y='averageRating',
                    color='startYear',
                    custom_data=['primaryTitle', 'genres', 'startYear'],
                    log_x=use_log_x,
                    title="Votes vs. Rating - Draw a box in upper-left to find hidden gems! 📦",
                    labels={'numVotes': 'Votes', 'averageRating': 'Rating'}
                )
                # Custom Tooltip
                fig.update_traces(
                    hovertemplate="<b>%{customdata[0]}</b><br>Rating: %{y}<br>Votes: %{x}<br>Year: %{customdata[2]}<extra></extra>"
                )
                fig.update_layout(hoverlabel=dict(bgcolor="white", font_size=14, font_family="sans-serif", font_color="black"))
                
                # Selection Event
                event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode="box")
                
                # Drill Down Table
                if event.selection["points"]:
                    selected_indices = [p["point_index"] for p in event.selection["points"]]
                    selected_movies = df_m_filtered.iloc[selected_indices]
                    
                    # Show results section
                    st.markdown(f"### ✨ You Found {len(selected_movies)} Hidden Gems!")
                    st.markdown("Here are your selected movies - perfect for your next movie night! 🍿")
                    
                    st.dataframe(
                        selected_movies[['primaryTitle', 'averageRating', 'numVotes', 'startYear']],
                        use_container_width=True,
                        column_config={
                            "primaryTitle": "Title",
                            "averageRating": st.column_config.NumberColumn("Rating", format="%.1f"),
                            "numVotes": "Votes",
                            "startYear": st.column_config.NumberColumn("Year", format="%d")
                        }
                    )
                    
                    # Add some fun stats about the selection
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        avg_rating = selected_movies['averageRating'].mean()
                        st.metric("⭐ Avg Rating", f"{avg_rating:.1f}")
                    with col2:
                        total_votes = selected_movies['numVotes'].sum()
                        st.metric("👥 Total Votes", f"{total_votes:,}")
                    with col3:
                        year_range = f"{selected_movies['startYear'].min()}-{selected_movies['startYear'].max()}"
                        st.metric("📅 Year Range", year_range)
                else:
                    # Show hint when no selection
                    st.info("👆 Draw a selection box around points in upper-left area to discover hidden gems!")

        with tab3:
            st.markdown("### Ratings over Time")
            if not df_m_filtered.empty:
                df_trend = df_m_filtered.groupby('startYear')['averageRating'].mean().reset_index()
                fig_line = px.line(df_trend, x='startYear', y='averageRating', title="Average Rating per Year")
                st.plotly_chart(fig_line, use_container_width=True)

        with tab4:
            st.markdown("### Interactive Timeline: Genre Popularity Over Time")
            
            # Prepare data for the genre timeline
            with st.spinner("Preparing genre timeline data..."):
                try:
                    genre_timeline = prepare_genre_timeline_data(df_movies, year_range[0], year_range[1])
                except Exception as e:
                    st.error(f"Error preparing genre timeline data: {str(e)}")
                    genre_timeline = pd.DataFrame()
            
            if genre_timeline.empty:
                st.warning("No genre timeline data available.")
            else:
                # Visualization options
                viz_type = st.radio("Visualization Type", ["Stacked Area Chart", "Line Chart", "Top Genres Over Time"])
                
                if viz_type == "Stacked Area Chart":
                    st.markdown("This chart shows percentage of movies in each genre over time. The total height represents 100% of movies released each year.")
                    
                    # Filter to show only top N genres to avoid overcrowding
                    top_n = st.slider("Number of Top Genres to Display", 5, 15, 10)
                    
                    # Calculate average percentage per genre across all years
                    avg_percentage = genre_timeline.groupby('genres')['percentage'].mean().reset_index()
                    top_genres = avg_percentage.sort_values('percentage', ascending=False)['genres'].head(top_n).tolist()
                    
                    # Filter data to include only top genres and "Other"
                    filtered_data = genre_timeline.copy()
                    filtered_data['genres'] = filtered_data['genres'].apply(
                        lambda x: x if x in top_genres else 'Other'
                    )
                    
                    # Aggregate "Other" category
                    other_data = filtered_data[filtered_data['genres'] == 'Other'].groupby(['startYear'])['percentage'].sum().reset_index()
                    other_data['genres'] = 'Other'
                    
                    # Get data for top genres
                    top_data = filtered_data[filtered_data['genres'] != 'Other'].groupby(['startYear', 'genres'])['percentage'].sum().reset_index()
                    
                    # Combine top genres and "Other"
                    final_data = pd.concat([top_data, other_data])
                    
                    # Create stacked area chart
                    fig = px.area(
                        final_data,
                        x='startYear',
                        y='percentage',
                        color='genres',
                        title=f"Top {top_n} Movie Genres Over Time ({year_range[0]}-{year_range[1]})",
                        labels={'startYear': 'Year', 'percentage': 'Percentage of Movies', 'genres': 'Genre'},
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    fig.update_layout(hovermode="x unified")
                    st.plotly_chart(fig, use_container_width=True)
                
                elif viz_type == "Line Chart":
                    st.markdown("This chart shows the absolute number of movies in each genre over time.")
                    
                    # Filter to show only top N genres to avoid overcrowding
                    top_n = st.slider("Number of Top Genres to Display", 5, 15, 10)
                    
                    # Calculate average count per genre across all years
                    avg_count = genre_timeline.groupby('genres')['count'].mean().reset_index()
                    top_genres = avg_count.sort_values('count', ascending=False)['genres'].head(top_n).tolist()
                    
                    # Filter data to include only top genres
                    filtered_data = genre_timeline[genre_timeline['genres'].isin(top_genres)]
                    
                    # Create line chart
                    fig = px.line(
                        filtered_data,
                        x='startYear',
                        y='count',
                        color='genres',
                        title=f"Top {top_n} Movie Genres by Count Over Time ({year_range[0]}-{year_range[1]})",
                        labels={'startYear': 'Year', 'count': 'Number of Movies', 'genres': 'Genre'},
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    fig.update_layout(hovermode="x unified")
                    st.plotly_chart(fig, use_container_width=True)
                
                else:  # Top Genres Over Time
                    st.markdown("This chart shows how the ranking of top genres has changed over time.")
                    
                    # Select number of top genres to track
                    top_n = st.slider("Number of Top Genres to Track", 3, 10, 5)
                    
                    # Get top genres by overall count
                    top_genres = genre_timeline.groupby('genres')['count'].sum().nlargest(top_n).index.tolist()
                    
                    # Filter data to include only top genres
                    filtered_data = genre_timeline[genre_timeline['genres'].isin(top_genres)]
                    
                    # Create ranking data
                    ranking_data = filtered_data.copy()
                    ranking_data['rank'] = ranking_data.groupby('startYear')['count'].rank(ascending=False)
                    
                    # Create bubble chart showing ranking over time
                    fig = px.scatter(
                        ranking_data,
                        x='startYear',
                        y='rank',
                        color='genres',
                        size='count',
                        title=f"Ranking of Top {top_n} Movie Genres Over Time ({year_range[0]}-{year_range[1]})",
                        labels={'startYear': 'Year', 'rank': 'Rank', 'count': 'Number of Movies', 'genres': 'Genre'},
                        color_discrete_sequence=px.colors.qualitative.Set3,
                        size_max=15
                    )
                    # Invert y-axis so rank 1 is at the top
                    fig.update_yaxes(autorange="reversed")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Genre details table
                with st.expander("View Genre Details"):
                    # Calculate genre statistics
                    genre_stats = genre_timeline.groupby('genres').agg({
                        'count': ['sum', 'mean'],
                        'percentage': 'mean'
                    }).reset_index()
                    genre_stats.columns = ['Genre', 'Total Movies', 'Avg Movies per Year', 'Avg Percentage']
                    genre_stats = genre_stats.sort_values('Total Movies', ascending=False)
                    
                    st.dataframe(
                        genre_stats,
                        use_container_width=True,
                        column_config={
                            "Genre": "Genre",
                            "Total Movies": st.column_config.NumberColumn("Total Movies", format="%d"),
                            "Avg Movies per Year": st.column_config.NumberColumn("Avg Movies per Year", format="%.1f"),
                            "Avg Percentage": st.column_config.NumberColumn("Avg Percentage", format="%.2f%%")
                        }
                    )

# ==========================================
# PAGE 2: TV SERIES ANALYZER
# ==========================================
elif mode == "TV Series Analyzer":
    # Back Button - show only when a series is selected
    if st.session_state.selected_show_tconst is not None:
        if st.button("⬅ Back to Search"):
            st.session_state.selected_show_tconst = None
            st.query_params.clear()
            st.rerun()
    
    st.header("📺 TV Series Analyzer")

    # --- MASTER VIEW (Search List) ---
    if st.session_state.selected_show_tconst is None:
        if df_series.empty:
            st.warning("No TV Series data found.")
        else:
            # 1. Filters
            c1, c2 = st.columns(2)
            with c1:
                all_genres_tv = set()
                for g in df_series['genres'].dropna():
                    all_genres_tv.update(g.split(','))
                sel_gen_tv = st.multiselect("Filter by Genre", sorted(list(all_genres_tv)))
            with c2:
                min_y_tv = int(df_series['startYear'].min()) if df_series['startYear'].notna().any() else 1900
                max_y_tv = int(df_series['startYear'].max()) if df_series['startYear'].notna().any() else 2025
                range_tv = st.slider("Filter by Year", min_y_tv, max_y_tv, (min_y_tv, max_y_tv))

            # 2. Apply Filters
            df_s_filtered = df_series.copy()
            if sel_gen_tv:
                mask = df_s_filtered['genres'].apply(lambda x: any(g in str(x).split(',') for g in sel_gen_tv))
                df_s_filtered = df_s_filtered[mask]
            
            df_s_filtered = df_s_filtered[
                (df_s_filtered['startYear'] >= range_tv[0]) & 
                (df_s_filtered['startYear'] <= range_tv[1])
            ]

            # 3. Search Bar
            search = st.text_input("Search Title...", "")
            if search:
                df_s_filtered = df_s_filtered[df_s_filtered['primaryTitle'].str.contains(search, case=False, na=False)]

            # 4. Info Box
            st.info(f"Showing {len(df_s_filtered)} series")

            # 5. Table with Link Column (NO checkboxes)
            df_s_display = df_s_filtered[['primaryTitle', 'startYear', 'averageRating', 'numVotes', 'genres', 'tconst']].copy()
            df_s_display = df_s_display.reset_index(drop=True)
            
            # Create clickable analyze links
            df_s_display['analyze_link'] = df_s_display['tconst'].apply(lambda x: f"/?tconst={x}")

            st.dataframe(
                df_s_display[['analyze_link', 'primaryTitle', 'startYear', 'averageRating', 'numVotes', 'genres']],
                use_container_width=True,
                height=600,
                hide_index=True,
                column_config={
                    "analyze_link": st.column_config.LinkColumn(
                        "Analyze",
                        display_text="🔍 Analyze"
                    ),
                    "primaryTitle": "Title",
                    "startYear": st.column_config.NumberColumn("Year", format="%d"),
                    "averageRating": st.column_config.NumberColumn("Rating", format="%.1f"),
                    "numVotes": "Votes",
                    "genres": "Genres"
                }
            )

    # --- DETAIL VIEW (Dashboard) ---
    else:
        # Get Info
        tconst = st.session_state.selected_show_tconst
        show_row = df_series[df_series['tconst'] == tconst]
        
        if show_row.empty:
            st.error("Error: Show details not found.")
        else:
            show_data = show_row.iloc[0]
            st.title(f"{show_data['primaryTitle']} ({show_data['startYear']})")
            st.markdown(f"**Rating:** ⭐ {show_data['averageRating']} | **Votes:** {show_data['numVotes']}")

            # Load Episodes
            with st.spinner("Loading episodes..."):
                df_ep = load_episode_data(tconst)
            
            if df_ep.empty:
                st.warning("No episode data available for this show.")
            else:
                # Clean Data
                df_ep = df_ep.dropna(subset=['seasonNumber', 'episodeNumber', 'averageRating'])
                df_ep['seasonNumber'] = df_ep['seasonNumber'].astype(int)
                df_ep['episodeNumber'] = df_ep['episodeNumber'].astype(int)

                # --- Tabs ---
                tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                    "🔥 Episode Heatmap", 
                    "📈 Season Trend", 
                    "📊 Rating Distribution", 
                    "⚖️ Season Comparison",
                    "🔍 Episode Explorer",
                    "📈 Series Evolution"
                ])

                with tab1:
                    st.subheader("Episode Heatmap")
                    
                    # Pivot
                    heatmap_data = df_ep.pivot_table(
                        index='episodeNumber', 
                        columns='seasonNumber', 
                        values='averageRating',
                        aggfunc='first'
                    ).sort_index(ascending=True)
                    
                    # Sort columns (seasons) just in case
                    heatmap_data = heatmap_data.reindex(sorted(heatmap_data.columns), axis=1)

                    # Custom Colors (Navy -> Red -> Orange -> Yellow -> Green)
                    colors = ['#00008B', '#D32F2F', '#FB8C00', '#FDD835', '#66BB6A', '#2E7D32']
                    boundaries = [0, 4, 6, 7, 8, 9, 10]
                    cmap = mcolors.ListedColormap(colors)
                    norm = mcolors.BoundaryNorm(boundaries, cmap.N)

                    fig, ax = plt.subplots(figsize=(12, 8))
                    sns.heatmap(
                        heatmap_data, cmap=cmap, norm=norm, annot=True, fmt=".1f",
                        linewidths=0.5, linecolor='gray', 
                        cbar_kws={'label': 'Rating'}, ax=ax
                    )
                    ax.set_xlabel("Season")
                    ax.set_ylabel("Episode")
                    # Move X axis to top for better readability
                    ax.xaxis.tick_top()
                    ax.xaxis.set_label_position('top')
                    st.pyplot(fig)

                with tab2:
                    st.subheader("Season Trend")
                    # Group by Season instead of Year (lite data optimization)
                    df_season = df_ep.groupby('seasonNumber')['averageRating'].mean().reset_index()
                    
                    fig_trend = px.line(
                        df_season, 
                        x='seasonNumber', 
                        y='averageRating', 
                        markers=True,
                        title="Average Rating per Season",
                        labels={'seasonNumber': 'Season', 'averageRating': 'Avg Rating'}
                    )
                    st.plotly_chart(fig_trend, use_container_width=True)

                with tab3:
                    st.subheader("Episode Rating Distribution")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Overall distribution
                        fig_hist = px.histogram(
                            df_ep, 
                            x='averageRating', 
                            nbins=20,
                            title="Overall Episode Rating Distribution",
                            labels={'averageRating': 'Rating', 'count': 'Number of Episodes'}
                        )
                        fig_hist.add_vline(
                            x=df_ep['averageRating'].mean(), 
                            line_dash="dash", 
                            annotation_text=f"Mean: {df_ep['averageRating'].mean():.2f}"
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)
                    
                    with col2:
                        # Box plot by season
                        fig_box = px.box(
                            df_ep, 
                            x='seasonNumber', 
                            y='averageRating',
                            title="Rating Distribution by Season",
                            labels={'seasonNumber': 'Season', 'averageRating': 'Rating'}
                        )
                        st.plotly_chart(fig_box, use_container_width=True)
                    
                    # Statistics
                    st.subheader("Rating Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Highest Rated", f"{df_ep['averageRating'].max():.1f}")
                    with col2:
                        st.metric("Lowest Rated", f"{df_ep['averageRating'].min():.1f}")
                    with col3:
                        st.metric("Average Rating", f"{df_ep['averageRating'].mean():.2f}")
                    with col4:
                        st.metric("Std Deviation", f"{df_ep['averageRating'].std():.2f}")

                with tab4:
                    st.subheader("Season Comparison")
                    
                    # Calculate season statistics
                    season_stats = df_ep.groupby('seasonNumber').agg({
                        'averageRating': ['mean', 'std', 'min', 'max', 'count'],
                        'episodeNumber': 'max'
                    }).round(2)
                    season_stats.columns = ['Avg Rating', 'Std Dev', 'Min Rating', 'Max Rating', 'Episode Count', 'Max Episode']
                    season_stats = season_stats.reset_index()
                    
                    # Display table
                    st.dataframe(
                        season_stats,
                        use_container_width=True,
                        column_config={
                            "seasonNumber": "Season",
                            "Avg Rating": st.column_config.NumberColumn("Avg Rating", format="%.2f"),
                            "Std Dev": st.column_config.NumberColumn("Std Dev", format="%.2f"),
                            "Min Rating": st.column_config.NumberColumn("Min Rating", format="%.1f"),
                            "Max Rating": st.column_config.NumberColumn("Max Rating", format="%.1f"),
                            "Episode Count": st.column_config.NumberColumn("Episodes", format="%d"),
                            "Max Episode": st.column_config.NumberColumn("Max Episode", format="%d")
                        }
                    )
                    
                    # Radar chart for season comparison
                    if len(season_stats) > 1:
                        # Normalize ratings for radar chart
                        radar_data = season_stats[['seasonNumber', 'Avg Rating', 'Max Rating']].copy()
                        radar_data['Norm Avg'] = (radar_data['Avg Rating'] - radar_data['Avg Rating'].min()) / (radar_data['Avg Rating'].max() - radar_data['Avg Rating'].min()) * 9 + 1
                        radar_data['Norm Max'] = (radar_data['Max Rating'] - radar_data['Max Rating'].min()) / (radar_data['Max Rating'].max() - radar_data['Max Rating'].min()) * 9 + 1
                        
                        fig_radar = px.line_polar(
                            radar_data, 
                            r='Norm Avg', 
                            theta='seasonNumber', 
                            line_close=True,
                            title="Season Comparison (Normalized Average Ratings)"
                        )
                        fig_radar.update_layout(polar_radialaxis_ticksuffix="")
                        st.plotly_chart(fig_radar, use_container_width=True)

                with tab5:
                    st.subheader("Episode Explorer")
                    
                    # Search and filter
                    col1, col2 = st.columns(2)
                    with col1:
                        season_filter = st.multiselect(
                            "Filter by Season", 
                            sorted(df_ep['seasonNumber'].unique()),
                            default=sorted(df_ep['seasonNumber'].unique())
                        )
                    with col2:
                        rating_range = st.slider(
                            "Rating Range", 
                            float(df_ep['averageRating'].min()), 
                            float(df_ep['averageRating'].max()),
                            (float(df_ep['averageRating'].min()), float(df_ep['averageRating'].max())),
                            step=0.1
                        )
                    
                    # Apply filters
                    df_filtered = df_ep[
                        (df_ep['seasonNumber'].isin(season_filter)) &
                        (df_ep['averageRating'] >= rating_range[0]) &
                        (df_ep['averageRating'] <= rating_range[1])
                    ].copy()
                    
                    # Sort options
                    sort_by = st.selectbox("Sort by", ["Rating (High to Low)", "Rating (Low to High)", "Season, Episode", "Votes"])
                    
                    if sort_by == "Rating (High to Low)":
                        df_filtered = df_filtered.sort_values('averageRating', ascending=False)
                    elif sort_by == "Rating (Low to High)":
                        df_filtered = df_filtered.sort_values('averageRating', ascending=True)
                    elif sort_by == "Season, Episode":
                        df_filtered = df_filtered.sort_values(['seasonNumber', 'episodeNumber'])
                    else:  # Votes
                        df_filtered = df_filtered.sort_values('numVotes', ascending=False)
                    
                    # Display episodes
                    st.dataframe(
                        df_filtered[['seasonNumber', 'episodeNumber', 'averageRating', 'numVotes']],
                        use_container_width=True,
                        column_config={
                            "seasonNumber": "Season",
                            "episodeNumber": "Episode",
                            "averageRating": st.column_config.NumberColumn("Rating", format="%.1f"),
                            "numVotes": "Votes"
                        }
                    )
                    
                    # Best and worst episodes
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("🏆 Best Episodes")
                        best_episodes = df_filtered.nlargest(5, 'averageRating')
                        st.dataframe(
                            best_episodes[['seasonNumber', 'episodeNumber', 'averageRating', 'numVotes']],
                            use_container_width=True,
                            column_config={
                                "seasonNumber": "Season",
                                "episodeNumber": "Episode",
                                "averageRating": st.column_config.NumberColumn("Rating", format="%.1f"),
                                "numVotes": "Votes"
                            }
                        )
                    
                    with col2:
                        st.subheader("👎 Worst Episodes")
                        worst_episodes = df_filtered.nsmallest(5, 'averageRating')
                        st.dataframe(
                            worst_episodes[['seasonNumber', 'episodeNumber', 'averageRating', 'numVotes']],
                            use_container_width=True,
                            column_config={
                                "seasonNumber": "Season",
                                "episodeNumber": "Episode",
                                "averageRating": st.column_config.NumberColumn("Rating", format="%.1f"),
                                "numVotes": "Votes"
                            }
                        )

                with tab6:
                    st.subheader("Series Evolution")
                    
                    # Create episode sequence number
                    df_ep_sorted = df_ep.sort_values(['seasonNumber', 'episodeNumber']).copy()
                    df_ep_sorted['episode_seq'] = range(1, len(df_ep_sorted) + 1)
                    
                    # Moving average
                    window_size = min(5, len(df_ep_sorted) // 4) if len(df_ep_sorted) > 4 else 1
                    df_ep_sorted['moving_avg'] = df_ep_sorted['averageRating'].rolling(window=window_size, center=True).mean()
                    
                    # Plot evolution
                    fig_evolution = px.line(
                        df_ep_sorted,
                        x='episode_seq',
                        y='averageRating',
                        color='seasonNumber',
                        title="Rating Evolution Throughout Series",
                        labels={
                            'episode_seq': 'Episode Number (Overall)',
                            'averageRating': 'Rating',
                            'seasonNumber': 'Season'
                        }
                    )
                    
                    # Add moving average line
                    fig_evolution.add_scatter(
                        x=df_ep_sorted['episode_seq'],
                        y=df_ep_sorted['moving_avg'],
                        mode='lines',
                        name=f'{window_size}-Episode Moving Average',
                        line=dict(dash='dash', color='black')
                    )
                    
                    st.plotly_chart(fig_evolution, use_container_width=True)
                    
                    # Consistency analysis
                    st.subheader("Rating Consistency Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Calculate consistency metrics
                        overall_std = df_ep['averageRating'].std()
                        season_std = df_ep.groupby('seasonNumber')['averageRating'].std().mean()
                        
                        st.metric("Overall Consistency (Std Dev)", f"{overall_std:.2f}")
                        st.metric("Average Season Consistency", f"{season_std:.2f}")
                        st.metric("Consistency Score", f"{max(0, 10 - overall_std):.1f}/10")
                    
                    with col2:
                        # Trend analysis
                        first_half = df_ep_sorted.iloc[:len(df_ep_sorted)//2]['averageRating'].mean()
                        second_half = df_ep_sorted.iloc[len(df_ep_sorted)//2:]['averageRating'].mean()
                        
                        trend = "Improving" if second_half > first_half else "Declining" if second_half < first_half else "Stable"
                        trend_color = "green" if trend == "Improving" else "red" if trend == "Declining" else "gray"
                        
                        st.markdown(f"**Trend:** <span style='color:{trend_color}'>{trend}</span>", unsafe_allow_html=True)
                        st.metric("First Half Avg", f"{first_half:.2f}")
                        st.metric("Second Half Avg", f"{second_half:.2f}")
                    
                    # Episode rating change detection
                    st.subheader("Significant Rating Changes")
                    
                    # Calculate episode-to-episode changes
                    df_ep_sorted['rating_change'] = df_ep_sorted['averageRating'].diff().abs()
                    
                    # Find significant changes (more than 1 standard deviation)
                    threshold = df_ep_sorted['rating_change'].std()
                    significant_changes = df_ep_sorted[df_ep_sorted['rating_change'] > threshold]
                    
                    if not significant_changes.empty:
                        st.dataframe(
                            significant_changes[['seasonNumber', 'episodeNumber', 'averageRating', 'rating_change']],
                            use_container_width=True,
                            column_config={
                                "seasonNumber": "Season",
                                "episodeNumber": "Episode",
                                "averageRating": st.column_config.NumberColumn("Rating", format="%.1f"),
                                "rating_change": st.column_config.NumberColumn("Rating Change", format="%.1f")
                            }
                        )
                    else:
                        st.info("No significant rating changes detected.")