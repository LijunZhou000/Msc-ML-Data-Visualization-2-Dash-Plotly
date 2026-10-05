from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html


DATA_DIR = Path(__file__).resolve().parent
FLIGHTS_FILE = DATA_DIR / "US_flights_2023.csv"
AIRPORTS_FILE = DATA_DIR / "airports.csv"
CHUNK_SIZE = 200_000
MIN_AIRPORT_FLIGHTS = 30
GROUP_COLUMNS = ["FlightDate", "State", "Dep_Airport"]
DELAY_COLUMNS = ["Dep_Delay", "Arr_Delay"]
TOTAL_DELAY = "Total_Delay"
DELAY_METRICS = [*DELAY_COLUMNS, TOTAL_DELAY]
CAUSE_COLUMNS = [
    "Delay_Carrier",
    "Delay_Weather",
    "Delay_NAS",
    "Delay_Security",
    "Delay_LastAircraft",
]
CAUSE_NAMES = {
    "Delay_Carrier": "Aerolínea",
    "Delay_Weather": "Meteorología",
    "Delay_NAS": "Sistema aéreo (NAS)",
    "Delay_Security": "Seguridad",
    "Delay_LastAircraft": "Avión anterior",
}
DELAY_NAMES = {
    "Dep_Delay": "salida",
    "Arr_Delay": "llegada",
    TOTAL_DELAY: "total (salida + llegada)",
}
STATE_NAMES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut",
    "DE": "Delaware", "DC": "Distrito de Columbia", "FL": "Florida",
    "GA": "Georgia", "HI": "Hawái", "ID": "Idaho", "IL": "Illinois",
    "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky",
    "LA": "Luisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Míchigan", "MN": "Minnesota",
    "MS": "Misisipi", "MO": "Misuri", "MT": "Montana",
    "NE": "Nebraska", "NV": "Nevada", "NH": "Nuevo Hampshire",
    "NJ": "Nueva Jersey", "NM": "Nuevo México", "NY": "Nueva York",
    "NC": "Carolina del Norte", "ND": "Dakota del Norte",
    "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregón",
    "PA": "Pensilvania", "RI": "Rhode Island",
    "SC": "Carolina del Sur", "SD": "Dakota del Sur",
    "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont",
    "VA": "Virginia", "WA": "Washington", "WV": "Virginia Occidental",
    "WI": "Wisconsin", "WY": "Wyoming", "PR": "Puerto Rico",
    "VI": "Islas Vírgenes de EE. UU.",
}


def load_data():
    """Aggregate the large flight file in chunks without losing exact medians."""
    required_flight_columns = [
        "FlightDate",
        "Dep_CityName",
        "Dep_Airport",
        "Arr_Airport",
        *DELAY_COLUMNS,
        *CAUSE_COLUMNS,
    ]
    flight_counts = []
    delay_histograms = []
    cause_counts = []

    for chunk in pd.read_csv(
        FLIGHTS_FILE,
        usecols=required_flight_columns,
        parse_dates=["FlightDate"],
        dtype={
            "Dep_CityName": "category",
            "Dep_Airport": "category",
            "Arr_Airport": "category",
        },
        chunksize=CHUNK_SIZE,
    ):
        for column in [*DELAY_COLUMNS, *CAUSE_COLUMNS]:
            chunk[column] = pd.to_numeric(chunk[column], errors="coerce").astype(
                "float32"
            )
        chunk["State"] = (
            chunk["Dep_CityName"]
            .astype("string")
            .str.rsplit(",", n=1)
            .str[-1]
            .str.strip()
            .fillna("Sin estado")
        )
        chunk.loc[chunk["State"].eq(""), "State"] = "Sin estado"
        chunk["Dep_Airport"] = (
            chunk["Dep_Airport"].astype("string").fillna("Sin código")
        )
        chunk["Arr_Airport"] = (
            chunk["Arr_Airport"].astype("string").fillna("Sin código")
        )
        chunk["FlightDate"] = pd.to_datetime(chunk["FlightDate"], errors="coerce")
        chunk = chunk.dropna(subset=["FlightDate"])
        if chunk.empty:
            continue

        flight_counts.append(
            chunk.groupby(
                GROUP_COLUMNS, as_index=False, observed=True, sort=False, dropna=False
            )
            .size()
            .rename(columns={"size": "flights"})
        )

        for delay_column in DELAY_COLUMNS:
            valid = chunk.loc[
                chunk[delay_column].notna(),
                GROUP_COLUMNS + ["Arr_Airport", delay_column],
            ].copy()
            if valid.empty:
                continue
            valid["Map_Airport"] = (
                valid["Dep_Airport"]
                if delay_column == "Dep_Delay"
                else valid["Arr_Airport"]
            )
            histogram = (
                valid.groupby(
                    GROUP_COLUMNS + ["Map_Airport", delay_column],
                    as_index=False,
                    observed=True,
                    sort=False,
                    dropna=False,
                )
                .size()
                .rename(columns={delay_column: "delay_minutes", "size": "flights"})
            )
            histogram["metric"] = delay_column
            delay_histograms.append(histogram)

            late = chunk.loc[
                chunk[delay_column].ge(15), GROUP_COLUMNS + CAUSE_COLUMNS
            ]
            if late.empty:
                continue
            cause_values = late[CAUSE_COLUMNS].fillna(0).to_numpy()
            dominant_indices = cause_values.argmax(axis=1)
            has_reported_cause = cause_values.max(axis=1) > 0
            if not has_reported_cause.any():
                continue
            attributed = late.loc[has_reported_cause, GROUP_COLUMNS].copy()
            attributed["cause"] = [
                CAUSE_COLUMNS[index] for index in dominant_indices[has_reported_cause]
            ]
            attributed["metric"] = delay_column
            cause_counts.append(
                attributed.groupby(
                    GROUP_COLUMNS + ["metric", "cause"],
                    as_index=False,
                    observed=True,
                    sort=False,
                    dropna=False,
                )
                .size()
                .rename(columns={"size": "flights"})
            )

        valid_total = chunk.loc[
            chunk[DELAY_COLUMNS].notna().all(axis=1),
            GROUP_COLUMNS + [*DELAY_COLUMNS, *CAUSE_COLUMNS],
        ].copy()
        if not valid_total.empty:
            valid_total["delay_minutes"] = (
                valid_total["Dep_Delay"] + valid_total["Arr_Delay"]
            )
            valid_total["Map_Airport"] = valid_total["Dep_Airport"]
            total_histogram = (
                valid_total.groupby(
                    GROUP_COLUMNS + ["Map_Airport", "delay_minutes"],
                    as_index=False,
                    observed=True,
                    sort=False,
                    dropna=False,
                )
                .size()
                .rename(columns={"size": "flights"})
            )
            total_histogram["metric"] = TOTAL_DELAY
            delay_histograms.append(total_histogram)

            late_total = valid_total.loc[valid_total["delay_minutes"].ge(15)]
            if not late_total.empty:
                cause_values = late_total[CAUSE_COLUMNS].fillna(0).to_numpy()
                dominant_indices = cause_values.argmax(axis=1)
                has_reported_cause = cause_values.max(axis=1) > 0
                if has_reported_cause.any():
                    attributed = late_total.loc[
                        has_reported_cause, GROUP_COLUMNS
                    ].copy()
                    attributed["cause"] = [
                        CAUSE_COLUMNS[index]
                        for index in dominant_indices[has_reported_cause]
                    ]
                    attributed["metric"] = TOTAL_DELAY
                    cause_counts.append(
                        attributed.groupby(
                            GROUP_COLUMNS + ["metric", "cause"],
                            as_index=False,
                            observed=True,
                            sort=False,
                            dropna=False,
                        )
                        .size()
                        .rename(columns={"size": "flights"})
                    )

    if not flight_counts:
        raise ValueError(f"No flight records were found in {FLIGHTS_FILE}")

    counts = (
        pd.concat(flight_counts, ignore_index=True)
        .groupby(GROUP_COLUMNS, as_index=False, observed=True, sort=False)["flights"]
        .sum()
    )
    if delay_histograms:
        histograms = (
            pd.concat(delay_histograms, ignore_index=True)
            .groupby(
                GROUP_COLUMNS + ["Map_Airport", "metric", "delay_minutes"],
                as_index=False,
                observed=True,
                sort=False,
            )["flights"]
            .sum()
        )
    else:
        histograms = pd.DataFrame(
            columns=GROUP_COLUMNS
            + ["Map_Airport", "metric", "delay_minutes", "flights"]
        )
    if cause_counts:
        causes = (
            pd.concat(cause_counts, ignore_index=True)
            .groupby(
                GROUP_COLUMNS + ["metric", "cause"],
                as_index=False,
                observed=True,
                sort=False,
            )["flights"]
            .sum()
        )
    else:
        causes = pd.DataFrame(
            columns=GROUP_COLUMNS + ["metric", "cause", "flights"]
        )

    airports = pd.read_csv(
        AIRPORTS_FILE,
        usecols=[
            "iata_code",
            "name",
            "municipality",
            "latitude_deg",
            "longitude_deg",
            "iso_country",
            "iso_region",
        ],
    )
    airports = airports.drop_duplicates("iata_code").copy()
    airports["latitude_deg"] = pd.to_numeric(
        airports["latitude_deg"], errors="coerce"
    )
    airports["longitude_deg"] = pd.to_numeric(
        airports["longitude_deg"], errors="coerce"
    )
    airports = airports[
        airports["iata_code"].isin(histograms["Map_Airport"].unique())
        | airports["iata_code"].isin(counts["Dep_Airport"].unique())
    ].copy()

    return counts, histograms, causes, airports


df_counts, df_histograms, df_causes, df_airports = load_data()
AVAILABLE_DATES = sorted(df_counts["FlightDate"].dt.strftime("%Y-%m-%d").unique())
FIRST_DATE = pd.Timestamp(AVAILABLE_DATES[0])
LAST_DATE = pd.Timestamp(AVAILABLE_DATES[-1])
STATE_CODES = sorted(df_counts["State"].dropna().unique())
AIRPORT_STATES = (
    df_counts.drop_duplicates("Dep_Airport")
    .set_index("Dep_Airport")["State"]
    .astype(str)
    .to_dict()
)
AIRPORT_DETAILS = df_airports.set_index("iata_code").to_dict("index")


def format_count(value):
    return f"{int(value):,}".replace(",", ".")


def airport_option(code):
    details = AIRPORT_DETAILS.get(code, {})
    city = details.get("municipality")
    label = f"{code} · {city}" if isinstance(city, str) and city.strip() else code
    return {"label": label, "value": code}


def airport_place(code):
    details = AIRPORT_DETAILS.get(code, {})
    city = details.get("municipality")
    region = details.get("iso_region")
    country = details.get("iso_country")
    place = city.strip() if isinstance(city, str) and city.strip() else ""

    if isinstance(region, str) and region.startswith("US-"):
        state_code = region.rsplit("-", 1)[-1]
        state = STATE_NAMES.get(state_code, state_code)
        return f"{place}, {state}" if place else state
    if isinstance(country, str) and country != "US":
        return f"{place}, {country}" if place else country
    return place or code


def filter_scope(frame, states, airports):
    filtered = frame
    if states:
        filtered = filtered[filtered["State"].isin(states)]
    if airports:
        filtered = filtered[filtered["Dep_Airport"].isin(airports)]
    return filtered


def weekly_data(frame, end_date, states, airports, days=7):
    end = pd.Timestamp(end_date)
    start = end - pd.Timedelta(days=days - 1)
    period = frame[frame["FlightDate"].between(start, end)]
    return filter_scope(period, states, airports)


def weighted_median(values, weights):
    if len(values) == 0 or int(np.sum(weights)) == 0:
        return None
    order = np.argsort(values)
    sorted_values = np.asarray(values)[order]
    cumulative = np.cumsum(np.asarray(weights)[order])
    total = int(cumulative[-1])
    lower = sorted_values[np.searchsorted(cumulative, (total - 1) // 2 + 1)]
    upper = sorted_values[np.searchsorted(cumulative, total // 2 + 1)]
    return float((lower + upper) / 2)


def weighted_mean(values, weights):
    total = int(np.sum(weights))
    if len(values) == 0 or total == 0:
        return None
    return float(np.average(values, weights=weights))


def delay_statistic(values, weights, statistic):
    if statistic == "mean":
        return weighted_mean(values, weights)
    return weighted_median(values, weights)


def summarize_period(end_date, states, airports, metric, statistic):
    period_counts = weekly_data(df_counts, end_date, states, airports)
    period_hist = weekly_data(df_histograms, end_date, states, airports)
    period_hist = period_hist[period_hist["metric"].eq(metric)]
    period_causes = weekly_data(df_causes, end_date, states, airports)
    period_causes = period_causes[period_causes["metric"].eq(metric)]

    flights = int(period_counts["flights"].sum())
    observed = int(period_hist["flights"].sum())
    late_flights = int(
        period_hist.loc[period_hist["delay_minutes"].ge(15), "flights"].sum()
    )
    delay_value = delay_statistic(
        period_hist["delay_minutes"].to_numpy(),
        period_hist["flights"].to_numpy(),
        statistic,
    )

    airport_stats = []
    for code, group in period_hist.groupby("Map_Airport", observed=True, sort=False):
        airport_flights = int(group["flights"].sum())
        if airport_flights < MIN_AIRPORT_FLIGHTS:
            continue
        airport_stats.append(
            (
                str(code),
                delay_statistic(
                    group["delay_minutes"].to_numpy(),
                    group["flights"].to_numpy(),
                    statistic,
                ),
                airport_flights,
            )
        )
    worst_airport = max(airport_stats, key=lambda row: row[1]) if airport_stats else None

    cause_totals = period_causes.groupby("cause", observed=True)["flights"].sum()
    dominant_cause = (
        str(cause_totals.idxmax()) if not cause_totals.empty else None
    )
    return {
        "flights": flights,
        "observed": observed,
        "late_flights": late_flights,
        "late_rate": late_flights / observed * 100 if observed else None,
        "delay_value": delay_value,
        "worst_airport": worst_airport,
        "causes": cause_totals,
        "cause_total": int(cause_totals.sum()),
        "cause_leader": dominant_cause,
    }


def format_minutes(value):
    return "—" if value is None else f"{value:+.1f} min"


def comparison_accent(current, previous):
    if current is None or previous is None:
        return "kpi-unchanged"
    difference = current - previous
    if abs(difference) < 0.05:
        return "kpi-unchanged"
    return "kpi-worsened" if difference > 0 else "kpi-improved"


def airport_state(code):
    details = AIRPORT_DETAILS.get(code, {})
    region = details.get("iso_region")
    if isinstance(region, str) and region.startswith("US-"):
        state_code = region.rsplit("-", 1)[-1]
        return STATE_NAMES.get(state_code, state_code)
    country = details.get("iso_country")
    if isinstance(country, str) and country:
        return country
    return "Sin estado"


def cause_percentage(summary):
    leader = summary.get("cause_leader")
    total = summary.get("cause_total", 0)
    if not leader or not total:
        return None
    return float(summary["causes"].get(leader, 0) / total * 100)


def make_kpi(title, value, detail, accent=None, show_dot=True):
    dot_colors = {
        "kpi-improved": "#35d07f",
        "kpi-worsened": "#ff4d5f",
        "kpi-unchanged": "#ffffff",
    }
    heading_children = []
    if show_dot:
        dot_class = accent or "kpi-unchanged"
        heading_children.append(
            html.Span(
                className=f"kpi-dot {dot_class}",
                style={
                    "backgroundColor": dot_colors.get(dot_class, "#ffffff"),
                    "display": "inline-block",
                    "width": "10px",
                    "height": "10px",
                    "minWidth": "10px",
                    "minHeight": "10px",
                    "borderRadius": "50%",
                    "boxShadow": "0 0 0 1px rgba(255,255,255,.22)",
                },
                **{"aria-hidden": "true"},
            )
        )
    heading_children.append(html.Span(title, className="kpi-title"))

    return html.Div(
        [
            html.Div(heading_children, className="kpi-heading"),
            html.Div(value, className="kpi-value"),
            html.Div(detail, className="kpi-detail"),
        ],
        className="kpi-card",
    )


def make_flight_summary(current, previous):
    return html.Div(
        [
            html.Div("VUELOS ANALIZADOS", className="section-label"),
            html.Div(
                [
                    html.Div(
                        [
                            html.Span("Últimos 7 días:", className="flight-summary-label"),
                            html.Span(format_count(current["flights"]), className="flight-summary-value"),
                            html.Span("vuelos", className="flight-summary-unit"),
                        ],
                        className="flight-summary-row",
                    ),
                    html.Div(
                        [
                            html.Span("7 días anteriores:", className="flight-summary-label"),
                            html.Span(format_count(previous["flights"]), className="flight-summary-value secondary"),
                            html.Span("vuelos", className="flight-summary-unit"),
                        ],
                        className="flight-summary-row",
                    ),
                ],
                className="flight-summary-values",
            ),
        ],
        className="flight-summary",
    )

def empty_figure(message):
    figure = go.Figure()
    figure.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"color": "#dce6f2", "size": 14},
    )
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
    )
    return figure


def style_figure(figure):
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Manrope, sans-serif", "color": "#dce6f2", "size": 11},
        margin={"l": 18, "r": 25, "t": 24, "b": 42},
        hoverlabel={"bgcolor": "#101c2c", "font": {"color": "#edf3fc"}},
    )
    return figure


def make_map(end_date, states, airports, metric, statistic):
    hist = weekly_data(df_histograms, end_date, states, airports)
    hist = hist[hist["metric"].eq(metric)]
    if hist.empty:
        return empty_figure("No hay retrasos informados para esta selección.")

    airport_rows = []
    for code, group in hist.groupby("Map_Airport", observed=True, sort=False):
        details = AIRPORT_DETAILS.get(str(code))
        if (
            details is None
            or pd.isna(details.get("longitude_deg"))
            or pd.isna(details.get("latitude_deg"))
        ):
            continue
        count = int(group["flights"].sum())
        airport_rows.append(
            {
                "code": str(code),
                "city": details.get("municipality") or "",
                "longitude": details["longitude_deg"],
                "latitude": details["latitude_deg"],
                "flights": count,
                "delay_stat": delay_statistic(
                    group["delay_minutes"].to_numpy(),
                    group["flights"].to_numpy(),
                    statistic,
                ),
            }
        )
    map_data = pd.DataFrame(airport_rows)
    if map_data.empty:
        return empty_figure(
            "No hay coordenadas de aeropuerto para representar esta selección."
        )

    sizes = 9 + 27 * np.sqrt(map_data["flights"] / map_data["flights"].max())
    figure = go.Figure(
        go.Scattergeo(
            lon=map_data["longitude"],
            lat=map_data["latitude"],
            mode="markers",
            marker={
                "size": sizes,
                "color": map_data["delay_stat"],
                "colorscale": [
                    [0, "#16803c"],
                    [0.25, "#ffffff"],
                    [1, "#c62828"],
                ],
                "cmin": -15,
                "cmax": 45,
                "colorbar": {
                    "title": {
                        "text": (
                            "Media del retraso" if statistic == "mean"
                            else "Mediana del retraso"
                        )
                    },
                    "tickvals": [-15, 0, 15, 30, 45],
                    "ticktext": [
                        "−15 · adelanto",
                        "0 · en hora",
                        "+15",
                        "+30",
                        "+45 · retraso",
                    ],
                    "tickfont": {"color": "#dce6f2"},
                    "outlinewidth": 0,
                },
                "line": {"color": "#09111f", "width": 1},
                "opacity": 0.9,
            },
            customdata=map_data[
                ["code", "city", "flights", "delay_stat"]
            ].to_numpy(dtype=object),
            hovertemplate=(
                "<b>%{customdata[0]}</b> · %{customdata[1]}"
                "<br>Vuelos con dato: %{customdata[2]:,}"
                f"<br>{'Media' if statistic == 'mean' else 'Mediana'}: "
                "%{customdata[3]:+.1f} min<extra></extra>"
            ),
            showlegend=False,
        )
    )
    figure.update_geos(
        scope="world",
        projection_type="natural earth",
        fitbounds="locations",
        showland=True,
        landcolor="#142237",
        showocean=True,
        oceancolor="#09111f",
        showcountries=True,
        countrycolor="#31445e",
        showsubunits=True,
        subunitcolor="#31445e",
        showlakes=False,
    )
    figure = style_figure(figure)
    figure.update_layout(
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        geo={"bgcolor": "rgba(0,0,0,0)"},
    )
    return figure


def make_bubble_legend(end_date, states, airports, metric):
    hist = weekly_data(df_histograms, end_date, states, airports)
    hist = hist[hist["metric"].eq(metric)]
    if hist.empty:
        return html.Div(
            "Tamaño de burbuja: vuelos con dato por aeropuerto.",
            className="bubble-legend",
        )

    airport_counts = hist.groupby("Map_Airport", observed=True)["flights"].sum()
    maximum = int(airport_counts.max())
    references = sorted({max(1, round(maximum * fraction)) for fraction in (0.25, 0.5, 1)})
    entries = []
    for count in references:
        diameter = 9 + 27 * np.sqrt(count / maximum)
        entries.append(
            html.Div(
                [
                    html.Span(
                        className="bubble-sample",
                        style={"width": f"{diameter:.1f}px", "height": f"{diameter:.1f}px"},
                        **{"aria-hidden": "true"},
                    ),
                    html.Span(f"{format_count(count)} vuelos"),
                ],
                className="bubble-legend-item",
                **{"aria-label": f"Burbuja de referencia: {format_count(count)} vuelos"},
            )
        )
    return html.Div(
        [
            html.Span("Tamaño · vuelos por aeropuerto:", className="bubble-legend-title"),
            *entries,
        ],
        className="bubble-legend",
        **{"aria-label": "Leyenda del tamaño de las burbujas"},
    )


def make_trend(end_date, states, airports, metric, statistic):
    end = pd.Timestamp(end_date)
    start = max(FIRST_DATE, end - pd.Timedelta(days=13))
    hist = df_histograms[
        df_histograms["FlightDate"].between(start, end)
        & df_histograms["metric"].eq(metric)
    ]
    hist = filter_scope(hist, states, airports)
    if hist.empty:
        return empty_figure("No hay datos para mostrar la tendencia.")

    daily_rows = []
    for day, group in hist.groupby("FlightDate", observed=True, sort=True):
        total = int(group["flights"].sum())
        delayed = int(
            group.loc[group["delay_minutes"].ge(15), "flights"].sum()
        )
        daily_rows.append(
            {
                "date": day,
                "delay_value": delay_statistic(
                    group["delay_minutes"].to_numpy(),
                    group["flights"].to_numpy(),
                    statistic,
                ),
                "late_rate": delayed / total * 100 if total else None,
                "observed": total,
            }
        )
    daily = pd.DataFrame(daily_rows)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=daily["date"],
            y=daily["delay_value"],
            mode="lines+markers",
            name=f"{'Media' if statistic == 'mean' else 'Mediana'} del retraso",
            line={"color": "#76aaff", "width": 3},
            marker={"size": 7},
            customdata=daily[["observed"]],
            hovertemplate=(
                f"%{{x|%d %b}}<br>{'Media' if statistic == 'mean' else 'Mediana'}: "
                "%{y:+.1f} min"
                "<br>Vuelos con dato: %{customdata[0]:,}<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=daily["date"],
            y=daily["late_rate"],
            mode="lines+markers",
            name="Vuelos con retraso ≥15 min",
            yaxis="y2",
            line={"color": "#ffbf69", "width": 2, "dash": "dot"},
            marker={"size": 6, "symbol": "square"},
            hovertemplate="%{x|%d %b}<br>Con retraso: %{y:.1f}%<extra></extra>",
        )
    )
    figure = style_figure(figure)
    figure.update_layout(
        xaxis={"title": "", "gridcolor": "rgba(142,161,185,.13)"},
        yaxis={
            "title": (
                "Media del retraso (min)" if statistic == "mean"
                else "Mediana del retraso (min)"
            ),
            "zeroline": True,
            "zerolinecolor": "rgba(142,161,185,.35)",
            "gridcolor": "rgba(142,161,185,.13)",
        },
        yaxis2={
            "title": "% con retraso ≥15 min",
            "overlaying": "y",
            "side": "right",
            "rangemode": "tozero",
            "ticksuffix": "%",
            "showgrid": False,
        },
        legend={"orientation": "h", "y": 1.12, "x": 0},
    )
    week_start = end - pd.Timedelta(days=6)
    figure.add_vrect(
        x0=week_start,
        x1=end + pd.Timedelta(hours=12),
        fillcolor="#61e3c1",
        opacity=0.07,
        line_width=0,
        annotation_text="Semana analizada",
        annotation_position="top left",
        annotation_font={"color": "#a8f0dc", "size": 10},
    )
    return figure


def make_causes_chart(summary):
    causes = summary["causes"].sort_values()
    if causes.empty:
        return empty_figure(
            "No hay causas registradas para vuelos retrasados ≥15 min."
        )
    percentages = causes / causes.sum() * 100
    labels = [CAUSE_NAMES.get(cause, cause) for cause in percentages.index]
    figure = go.Figure(
        go.Bar(
            x=percentages.to_numpy(),
            y=labels,
            orientation="h",
            marker={"color": "#ffbf69"},
            text=[f"{value:.1f}%" for value in percentages],
            textposition="outside",
            cliponaxis=False,
            customdata=causes.to_numpy(),
            hovertemplate=(
                "<b>%{y}</b><br>%{x:.1f}% de los casos atribuidos"
                "<br>Vuelos: %{customdata:,}<extra></extra>"
            ),
        )
    )
    figure = style_figure(figure)
    figure.update_layout(
        xaxis={
            "title": "Parte de los vuelos con causa identificada",
            "ticksuffix": "%",
            "rangemode": "tozero",
            "gridcolor": "rgba(142,161,185,.13)",
        },
        yaxis={"title": "", "automargin": True},
        margin={"l": 20, "r": 50, "t": 18, "b": 45},
        showlegend=False,
    )
    return figure


app = Dash(__name__)
app.title = "Flight Pulse | Retrasos de vuelos"
app.index_string = """
<!DOCTYPE html>
<html lang="es">
    <head>
        {%metas%}
        <title>Flight Pulse | Retrasos de vuelos</title>
        {%favicon%}
        {%css%}
        <style>
            :root { color-scheme: dark; font-family: Arial, sans-serif; background: #09111f; color: #edf3fc; }
            * { box-sizing: border-box; }
            body { margin: 0; min-width: 320px; background: #09111f; color: #edf3fc; }
            :focus-visible { outline: 3px solid #fbbf24 !important; outline-offset: 3px; }
            .app-shell {
                display: grid; grid-template-columns: minmax(230px, 280px) minmax(0, 1fr);
                align-items: start; column-gap: 20px; max-width: 1600px;
                margin: auto; padding: 34px 28px 48px;
            }
            .app-shell > .hero { grid-column: 1 / -1; }
            .app-shell > .filter-panel {
                grid-column: 1; grid-row: 2; position: sticky; top: 16px; z-index: 5;
                align-self: start; max-height: calc(100vh - 32px); overflow-y: auto;
                margin-bottom: 0;
            }
            .app-shell > :not(.hero):not(.filter-panel) { grid-column: 2; min-width: 0; }
            .hero { display: flex; justify-content: space-between; align-items: flex-end; gap: 20px; margin-bottom: 22px; }
            .eyebrow, .section-label { color: #75e2c4; font-size: 11px; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
            .hero h1 { margin: 8px 0; font-size: clamp(32px, 5vw, 52px); letter-spacing: -.04em; }
            .hero p, .panel p { color: #b7c5d8; line-height: 1.5; }
            .hero p { max-width: 720px; margin: 0; }
            .hero-badge { padding: 10px 14px; border: 1px solid #38516d; border-radius: 10px; color: #dce6f2; background: #101c2c; white-space: nowrap; }
            .panel { border: 1px solid #2b3d54; border-radius: 16px; background: #101c2c; }
            .filter-panel { padding: 18px; margin-bottom: 18px; }
            .filters { display: grid; grid-template-columns: minmax(0, 1fr); gap: 14px; align-items: end; margin-top: 14px; }
            .filter-label { display: block; margin-bottom: 7px; color: #dce6f2; font-size: 12px; font-weight: 700; }
            .filter-help { margin: 7px 0 0; color: #b7c5d8; font-size: 12px; line-height: 1.45; }
            .DateInput, .DateInput_input, .SingleDatePicker, .SingleDatePickerInput { width: 100%; }
            .DateInput, .DateInput_input, .DateInput_input__focused { background-color: #09111f !important; }
            .DateInput { border: 1px solid #52657d; border-radius: 8px; }
            .DateInput_input {
                padding: 10px; border: 0 !important; border-radius: 8px;
                color: #c3cfdd !important; -webkit-text-fill-color: #c3cfdd !important;
                background-color: #09111f !important; font-size: 14px;
            }
            .SingleDatePickerInput { border: 0; background: transparent; }
            .DateInput_fang { display: none; }
            .filter-dropdown .Select-control { min-height: 42px; border: 1px solid #52657d; border-radius: 8px; background: #09111f; color: #edf3fc; }
            .filter-dropdown .Select-placeholder, .filter-dropdown .Select-value-label, .filter-dropdown .Select-input > input { color: #edf3fc !important; }
            .filter-dropdown .Select-menu-outer { z-index: 20; border: 1px solid #52657d; background: #101c2c; color: #edf3fc; }
            .filter-dropdown .VirtualizedSelectOption { color: #edf3fc; background: #101c2c; }
            .filter-dropdown .VirtualizedSelectFocusedOption { background: #29435f; }
            .filter-dropdown .Select--multi .Select-value { border: 1px solid #397366; border-radius: 5px; background: #15312f; color: #d6fff3; }
            .filter-radio label { margin-right: 16px; color: #edf3fc; font-size: 14px; cursor: pointer; }
            .filter-radio input { margin-right: 6px; accent-color: #61e3c1; }
            .flight-summary { margin-bottom: 20px; padding-bottom: 18px; border-bottom: 1px solid #2b3d54; }
            .flight-summary-values { display: grid; grid-template-columns: 1fr; gap: 8px; margin-top: 10px; }
            .flight-summary-row { display: flex; align-items: baseline; gap: 6px; min-width: 0; }
            .flight-summary-value { color: #fff; font-size: 24px; font-weight: 800; line-height: 1.05; letter-spacing: -.02em; }
            .flight-summary-value.secondary { color: #dce6f2; font-size: 22px; }
            .flight-summary-label { color: #b7c5d8; font-size: 12px; line-height: 1.25; white-space: nowrap; }
            .flight-summary-unit { color: #b7c5d8; font-size: 11px; }
            .kpi-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 13px; margin-bottom: 13px; }
            .kpi-grid.kpi-grid-secondary { grid-template-columns: minmax(0, 2fr) minmax(0, 1fr) !important; }
            .kpi-card { min-width: 0; min-height: 132px; padding: 17px; border: 1px solid #2b3d54; border-radius: 13px; background: #101c2c; }
            .kpi-heading { display: flex; align-items: center; gap: 8px; }
            .kpi-title { color: #c3cfdd; font-size: 11px; font-weight: 700; letter-spacing: .04em; text-transform: uppercase; }
            .kpi-dot { width: 9px; height: 9px; flex: 0 0 9px; border-radius: 50%; }
            .kpi-improved { background: #35d07f; }
            .kpi-worsened { background: #ff4d5f; }
            .kpi-unchanged { background: #ffffff; }
            .kpi-value { overflow-wrap: anywhere; margin: 15px 0 6px; color: #fff; font-size: clamp(21px, 2.5vw, 31px); font-weight: 800; line-height: 1.08; }
            .kpi-detail { color: #b7c5d8; font-size: 12px; line-height: 1.4; }
            .kpi-airport-value { font-size: clamp(18px, 2.15vw, 28px); }
            .map-panel, .chart-panel, .data-panel { padding: 18px; }
            .map-header { display: flex; justify-content: space-between; align-items: flex-start; gap: 18px; }
            .map-header h2, .chart-panel h2, .data-panel h2 { margin: 5px 0; font-size: 19px; }
            .map-header p, .chart-panel p { margin: 5px 0; font-size: 13px; }
            .map-panel .section-label, .chart-panel .section-label { margin-bottom: 7px; }
            .bubble-legend { display: flex; flex-wrap: wrap; align-items: center; gap: 9px 18px; padding: 5px 8px 12px; color: #dce6f2; font-size: 12px; }
            .bubble-legend-title { font-weight: 700; }
            .bubble-legend-item { display: inline-flex; align-items: center; gap: 7px; min-height: 34px; }
            .bubble-sample { display: inline-block; flex: 0 0 auto; border: 2px solid #dce6f2; border-radius: 50%; background: rgba(220,230,242,.18); }
            .insights-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; margin-top: 16px; }
            .data-panel { margin-top: 16px; }
            .data-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; }
            .data-grid p { margin: 6px 0 0; font-size: 13px; }
            .data-grid strong { color: #fff; }
            .footer-note { margin-top: 15px; color: #b7c5d8; text-align: right; font-size: 11px; }
            @media (max-width: 900px) { .insights-grid { grid-template-columns: 1fr; } }
            @media (max-width: 700px) {
                .app-shell { grid-template-columns: minmax(0, 1fr); column-gap: 0; }
                .app-shell > .hero, .app-shell > .filter-panel,
                .app-shell > :not(.hero):not(.filter-panel) { grid-column: 1; }
                .app-shell > .filter-panel {
                    grid-row: auto; position: static; max-height: none; overflow: visible;
                    margin-bottom: 18px;
                }
                .filters { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            }
            @media (max-width: 600px) {
                .app-shell { padding: 22px 14px 32px; }
                .hero { align-items: flex-start; flex-direction: column; }
                .filters, .kpi-grid, .data-grid { grid-template-columns: 1fr; }
                .kpi-grid.kpi-grid-secondary { grid-template-columns: 1fr !important; }
                .map-header { flex-direction: column; }
                .kpi-card { min-height: 118px; }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""

app.layout = html.Main(
    [
        html.Header(
            [
                html.Div(
                    [
                        html.Div(
                            "Demo Day 2 · Aviación estadounidense · 2023",
                            className="eyebrow",
                        ),
                        html.H1("Flight Pulse"),
                        html.P(
                            "Para responsables de operaciones: ¿dónde conviene priorizar "
                            "el seguimiento? Compara retrasos, aeropuertos y causas "
                            "declaradas semana a semana."
                        ),
                    ]
                ),
                html.Div("DATOS DE VUELOS 2023", className="hero-badge"),
            ],
            className="hero",
        ),
        html.Section(
            [
                html.Div(id="vuelos-resumen"),
                html.Div("FILTRA LA OPERACIÓN", className="section-label"),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Label(
                                    "Último día de la semana",
                                    htmlFor="dia",
                                    className="filter-label",
                                ),
                                dcc.DatePickerSingle(
                                    id="dia",
                                    date=AVAILABLE_DATES[-1],
                                    min_date_allowed=AVAILABLE_DATES[0],
                                    max_date_allowed=AVAILABLE_DATES[-1],
                                    initial_visible_month=AVAILABLE_DATES[-1],
                                    display_format="DD/MM/YYYY",
                                    number_of_months_shown=1,
                                    clearable=False,
                                ),
                            ]
                        ),
                        html.Div(
                            [
                                html.Label(
                                    "Estado",
                                    htmlFor="estado",
                                    className="filter-label",
                                ),
                                dcc.Dropdown(
                                    id="estado",
                                    options=[
                                        {
                                            "label": (
                                                f"{STATE_NAMES.get(code, code)} ({code})"
                                            ),
                                            "value": code,
                                        }
                                        for code in STATE_CODES
                                    ],
                                    value=[],
                                    multi=True,
                                    placeholder="Todos los estados",
                                    className="filter-dropdown",
                                ),
                            ]
                        ),
                        html.Div(
                            [
                                html.Label(
                                    "Aeropuerto de salida",
                                    htmlFor="aeropuerto",
                                    className="filter-label",
                                ),
                                dcc.Dropdown(
                                    id="aeropuerto",
                                    options=[],
                                    value=[],
                                    multi=True,
                                    disabled=True,
                                    placeholder="Elige primero un estado",
                                    className="filter-dropdown",
                                ),
                            ]
                        ),
                        html.Div(
                            [
                                html.Label(
                                    "Retraso analizado",
                                    htmlFor="retraso",
                                    className="filter-label",
                                ),
                                dcc.RadioItems(
                                    id="retraso",
                                    value=TOTAL_DELAY,
                                    options=[
                                        {"label": "Total", "value": TOTAL_DELAY},
                                        {"label": "Salida", "value": "Dep_Delay"},
                                        {"label": "Llegada", "value": "Arr_Delay"},
                                    ],
                                    className="filter-radio",
                                    inputStyle={"marginLeft": "0", "marginRight": "6px"},
                                    labelStyle={"display": "inline-block"},
                                ),
                                html.P(
                                    "Total suma el retraso de salida y el de llegada "
                                    "del mismo vuelo; solo incluye vuelos con ambos datos.",
                                    className="filter-help",
                                ),
                                html.Label(
                                    "Métrica de resumen",
                                    htmlFor="estadistico",
                                    className="filter-label",
                                ),
                                dcc.RadioItems(
                                    id="estadistico",
                                    value="median",
                                    options=[
                                        {"label": "Mediana", "value": "median"},
                                        {"label": "Media", "value": "mean"},
                                    ],
                                    className="filter-radio",
                                    inputStyle={"marginLeft": "0", "marginRight": "6px"},
                                    labelStyle={"display": "inline-block"},
                                ),
                            ]
                        ),
                    ],
                    className="filters",
                ),
            ],
            className="panel filter-panel",
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(id="kpi-retrasados"),
                                html.Div(id="kpi-porcentaje"),
                                html.Div(id="kpi-mediana"),
                            ],
                            className="kpi-grid",
                        ),
                        html.Div(
                            [
                                html.Div(id="kpi-aeropuerto"),
                                html.Div(id="kpi-causa"),
                            ],
                            className="kpi-grid kpi-grid-secondary",
                        ),
                    ],
                    **{"aria-label": "Cinco indicadores clave de retrasos"},
                ),
                html.Section(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div("DISTRIBUCIÓN GEOGRÁFICA", className="section-label"),
                                        html.H2(
                                            "Retraso mediano por aeropuerto",
                                            id="map-title",
                                        ),
                                        html.P(
                                            id="map-description",
                                            children=(
                                                "El color representa la mediana del retraso; "
                                                "el tamaño indica el volumen de vuelos."
                                            ),
                                        ),
                                    ]
                                ),
                            ],
                            className="map-header",
                        ),
                        dcc.Graph(
                            id="mapa",
                            config={"displayModeBar": True, "scrollZoom": True, "responsive": True},
                            style={"height": "min(58vh, 520px)", "minHeight": "350px"},
                        ),
                        html.Div(id="leyenda-burbujas"),
                    ],
                    className="panel map-panel",
                ),
                html.Div(
                    [
                        html.Section(
                            [
                                html.Div("EVOLUCIÓN", className="section-label"),
                                html.H2("¿Los retrasos mejoran o empeoran?"),
                                html.P(
                                    "Media o mediana diaria según el selector y proporción de vuelos "
                                    "con retraso ≥15 min; "
                                    "la franja clara marca la semana analizada."
                                ),
                                dcc.Graph(
                                    id="grafico-tendencia",
                                    config={"displayModeBar": True, "responsive": True},
                                    style={"height": "350px"},
                                ),
                            ],
                            className="panel chart-panel",
                        ),
                        html.Section(
                            [
                                html.Div("CAUSAS DECLARADAS", className="section-label"),
                                html.H2("¿Qué causas aparecen con más frecuencia?"),
                                html.P(
                                    "En vuelos con retraso ≥15 min, cada vuelo se cuenta una sola vez "
                                    "según la categoría declarada con mayor valor."
                                ),
                                dcc.Graph(
                                    id="grafico-causas",
                                    config={"displayModeBar": True, "responsive": True},
                                    style={"height": "350px"},
                                ),
                            ],
                            className="panel chart-panel",
                        ),
                    ],
                    className="insights-grid",
                ),
                html.Section(
                    [
                        html.Div("DATOS Y CALIDAD", className="section-label"),
                        html.H2("Qué representa este panel"),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Strong("Fuente"),
                                        html.P(
                                            "US_flights_2023.csv para vuelos y causas; airports.csv "
                                            "para coordenadas y nombres de aeropuertos."
                                        ),
                                    ]
                                ),
                                html.Div(
                                    [
                                        html.Strong("Grano"),
                                        html.P(
                                            "Una fila del fichero de vuelos representa una operación. "
                                            "Las comparaciones agrupan por semana, estado y aeropuerto "
                                            "de salida."
                                        ),
                                    ]
                                ),
                                html.Div(
                                    [
                                        html.Strong("Calidad y límites"),
                                        html.P(
                                            "Los retrasos ausentes o no numéricos se excluyen de "
                                            "medias, medianas y tasas; las fechas no válidas se excluyen de "
                                            "los totales. La cobertura de retrasos se muestra en el "
                                            "detalle de los KPI. Sin coordenadas, un vuelo no aparece "
                                            "en el mapa, pero sí en los totales."
                                        ),
                                    ]
                                ),
                            ],
                            className="data-grid",
                        ),
                    ],
                    className="panel data-panel",
                ),
                html.Div(
                    "Cada KPI compara los 7 días que terminan en la fecha elegida con los 7 días anteriores.",
                    className="footer-note",
                ),
            ],
            className="dashboard-content",
        ),
    ],
    className="app-shell",
)


@app.callback(
    Output("aeropuerto", "options"),
    Output("aeropuerto", "value"),
    Output("aeropuerto", "disabled"),
    Output("aeropuerto", "placeholder"),
    Input("estado", "value"),
    State("aeropuerto", "value"),
)
def update_airport_options(states, selected_airports):
    if not states:
        return [], [], True, "Elige primero un estado"
    codes = sorted(
        airport
        for airport, state in AIRPORT_STATES.items()
        if state in states and airport in AIRPORT_DETAILS
    )
    retained = [airport for airport in (selected_airports or []) if airport in codes]
    return (
        [airport_option(code) for code in codes],
        retained,
        False,
        "Todos los aeropuertos de los estados elegidos",
    )


@app.callback(
    Output("vuelos-resumen", "children"),
    Output("kpi-retrasados", "children"),
    Output("kpi-porcentaje", "children"),
    Output("kpi-mediana", "children"),
    Output("kpi-aeropuerto", "children"),
    Output("kpi-causa", "children"),
    Output("mapa", "figure"),
    Output("grafico-tendencia", "figure"),
    Output("grafico-causas", "figure"),
    Output("map-description", "children"),
    Output("map-title", "children"),
    Output("leyenda-burbujas", "children"),
    Input("dia", "date"),
    Input("estado", "value"),
    Input("aeropuerto", "value"),
    Input("retraso", "value"),
    Input("estadistico", "value"),
)
def update_dashboard(selected_date, states, airports, metric, statistic):
    end_date = pd.Timestamp(selected_date or LAST_DATE)
    states = states or []
    airports = airports or []
    metric = metric if metric in DELAY_NAMES else TOTAL_DELAY
    statistic = statistic if statistic in {"median", "mean"} else "median"
    statistic_name = "Media" if statistic == "mean" else "Mediana"
    current = summarize_period(end_date, states, airports, metric, statistic)
    previous = summarize_period(
        end_date - pd.Timedelta(days=7), states, airports, metric, statistic
    )
    delay_name = DELAY_NAMES[metric]

    current_airport = (
        f"{current['worst_airport'][0]} · {airport_state(current['worst_airport'][0])} · "
        f"{current['worst_airport'][1]:+.1f} min"
        if current["worst_airport"]
        else "—"
    )
    previous_airport = (
        f"{previous['worst_airport'][0]} · {airport_state(previous['worst_airport'][0])} · "
        f"{previous['worst_airport'][1]:+.1f} min"
        if previous["worst_airport"]
        else "—"
    )
    previous_cause = (
        CAUSE_NAMES.get(previous["cause_leader"], previous["cause_leader"])
        if previous["cause_leader"]
        else "—"
    )
    current_cause = (
        CAUSE_NAMES.get(current["cause_leader"], current["cause_leader"])
        if current["cause_leader"]
        else "—"
    )
    current_cause_pct = cause_percentage(current)
    previous_cause_pct = cause_percentage(previous)

    kpis = [
        make_kpi(
            "Vuelos con retraso ≥15 min",
            format_count(current["late_flights"]),
            f"Semana anterior: {format_count(previous['late_flights'])}",
            comparison_accent(current["late_flights"], previous["late_flights"]),
        ),
        make_kpi(
            "Tasa de vuelos con retraso ≥15 min",
            "—" if current["late_rate"] is None else f"{current['late_rate']:.1f}%",
            "Semana anterior: —" if previous["late_rate"] is None else f"Semana anterior: {previous['late_rate']:.1f}%",
            comparison_accent(current["late_rate"], previous["late_rate"]),
        ),
        make_kpi(
            f"{statistic_name} del retraso de {delay_name}",
            format_minutes(current["delay_value"]),
            (
                "Semana anterior: —"
                if previous["delay_value"] is None
                else f"Semana anterior: {format_minutes(previous['delay_value'])}"
            ),
            comparison_accent(current["delay_value"], previous["delay_value"]),
        ),
        make_kpi(
            f"Aeropuerto con mayor {statistic_name.lower()} de {delay_name}",
            current_airport,
            f"Semana anterior: {previous_airport}",
            show_dot=False,
        ),
        make_kpi(
            "Causa declarada más frecuente",
            (
                f"{current_cause} · {current_cause_pct:.1f}%"
                if current_cause_pct is not None and current_cause != "—"
                else current_cause
            ),
            (
                f"Semana anterior: {previous_cause} · {previous_cause_pct:.1f}%"
                if previous_cause_pct is not None and previous_cause != "—"
                else f"Semana anterior: {previous_cause}"
            ),
            show_dot=False,
        ),
    ]

    return (
        make_flight_summary(current, previous),
        *kpis,
        make_map(end_date, states, airports, metric, statistic),
        make_trend(end_date, states, airports, metric, statistic),
        make_causes_chart(current),
        (
            f"El color representa la {statistic_name.lower()} del retraso por vuelo; "
            if metric != TOTAL_DELAY
            else (
                f"El color representa la {statistic_name.lower()} del retraso total por vuelo "
                "(salida + llegada); "
            )
        )
        + (
            "Cada burbuja representa un aeropuerto de salida."
            if metric in {"Dep_Delay", TOTAL_DELAY}
            else (
                "Cada burbuja representa un aeropuerto de destino de los vuelos "
                "que parten de los estados y aeropuertos seleccionados."
            )
        )
        + " Rojo: retraso; blanco: en hora; verde: adelanto. "
        + "La escala de color se mantiene fija para comparar semanas.",
        f"{statistic_name} del retraso por aeropuerto",
        make_bubble_legend(end_date, states, airports, metric),
    )


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)