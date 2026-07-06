WITH base_metrics AS (
    SELECT 
        game_id,
        model_name,
        turn_number,
        nearest_gold_distance,
        response_time_sec,
        is_invalid_move,
        game_outcome,
        AVG(nearest_gold_distance) OVER (
            PARTITION BY game_id ORDER BY turn_number ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ) as moving_avg_gold_distance
    FROM {{ ref('stg_simulation_logs') }}
)
SELECT 
    game_id,
    model_name,
    MAX(turn_number) as total_turns,
    AVG(response_time_sec) as avg_response_time_sec,
    SUM(CASE WHEN is_invalid_move THEN 1 ELSE 0 END) as total_invalid_moves,
    MAX(CASE WHEN game_outcome = 'VICTOIRE' THEN 1 ELSE 0 END) as is_success,
    AVG(moving_avg_gold_distance) as global_avg_distance
FROM base_metrics
GROUP BY game_id, model_name