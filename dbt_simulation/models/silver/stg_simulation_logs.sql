SELECT
    cast(game_id as VARCHAR) as game_id,
    cast(run_timestamp as TIMESTAMP) as run_timestamp,
    cast(model_name as VARCHAR) as model_name,
    cast(turn_number as INTEGER) as turn_number,
    cast(player_row as INTEGER) as player_row,
    cast(player_col as INTEGER) as player_col,
    cast(golds_count as INTEGER) as golds_count,
    cast(nearest_gold_distance as DOUBLE) as nearest_gold_distance,
    cast(enemies_count as INTEGER) as enemies_count,
    cast(llm_decision as VARCHAR) as llm_decision,
    cast(response_time_sec as DOUBLE) as response_time_sec,
    cast(is_invalid_move as BOOLEAN) as is_invalid_move,
    cast(game_outcome as VARCHAR) as game_outcome
FROM {{ source('simulation_bronze', 'raw_logs') }}