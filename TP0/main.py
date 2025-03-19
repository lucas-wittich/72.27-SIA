import json
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.catching import attempt_catch
from src.pokemon import PokemonFactory, StatusEffect, Pokemon

# Initialize DataFrame to store results
pokemon_stats = pd.DataFrame(columns=['Pokemon', 'Pokeball', 'HP%', 'Status', 'Capture_Success',
                             'Capture_Probability', 'Noise', 'Catch_Rate', 'Level', 'Weight', 'Type'])
noise_levels = [0.0, 0.05, 0.1, 0.15]
pokemon_configs = ['caterpie', 'snorlax']
success_rates = {}


def simulate_attempts(pokemon: Pokemon, pokeball: str, attempts: int = 100, noise: float = 0, catch_rate: float = 0, weight: int = 0, pokemon_type=''):
    results = []
    for _ in range(attempts):
        success, probability = attempt_catch(pokemon, pokeball, noise)
        # print(f'Debug: {pokemon.name} | Pokeball: {pokeball} | HP%: {pokemon.current_hp} | Success: {success} | Probability: {probability}')
        results.append({
            'Pokemon': pokemon.name,
            'Pokeball': pokeball,
            'HP%': round((pokemon.current_hp / pokemon.max_hp) * 100),
            'Status': pokemon.status_effect.name,
            'Capture_Success': success,
            'Capture_Probability': probability,
            'Noise': noise,
            'Catch_Rate': catch_rate,
            'Level': pokemon.level,
            'Weight': weight,
            'Type': pokemon_type
        })
    return results


if __name__ == '__main__':
    factory = PokemonFactory('pokemon.json')

    # Load all Pokemon data
    with open('pokemon.json', 'r') as f:
        pokemon_data = json.load(f)

    # Iterate over Pokemon configurations
    for pokemon_config in pokemon_configs:
        with open(f'configs/{pokemon_config}.json', 'r') as f:
            config = json.load(f)
            ball = config['pokeball']
            pokemon_name = config['pokemon']

            # Extract Pokemon attributes from pokemon.json
            catch_rate = pokemon_data[pokemon_name]['catch_rate']
            weight = pokemon_data[pokemon_name]['weight']
            pokemon_type = ', '.join(pokemon_data[pokemon_name]['type'])

            for level in [50, 100, 150]:
                for status in [StatusEffect.NONE, StatusEffect.SLEEP, StatusEffect.PARALYSIS]:
                    for hp in range(100, 0, -10):
                        for noise in noise_levels:
                            pokemon = factory.create(
                                pokemon_name, level, status, hp / 100)
                            results = simulate_attempts(
                                pokemon, ball, 10, noise, catch_rate, weight, pokemon_type)
                            df = pd.DataFrame(results)
                            pokemon_stats = pd.concat(
                                [pokemon_stats, df], ignore_index=True)

    # Compute capture success rates
    pokemon_stats['Capture_Success'] = pokemon_stats['Capture_Success'].astype(
        int)
    success_rates = pokemon_stats.groupby('Pokeball')['Capture_Success'].mean()
    summary_df = pokemon_stats.groupby(['Pokemon', 'Pokeball', 'HP%', 'Status', 'Noise', 'Catch_Rate', 'Weight', 'Type']).agg(
        Success_Rate=('Capture_Success', 'mean'),  # Mean success rate
        Avg_Capture_Probability=(
            'Capture_Probability', 'mean'),  # Mean probability
        # Standard deviation (optional)
        Std_Capture_Probability=('Capture_Probability', 'std'),
        Attempts=('Capture_Success', 'count')  # Number of attempts
    ).reset_index()
    summary_df = summary_df.round(
        {'Success_Rate': 2, 'Avg_Capture_Probability': 2, 'Std_Capture_Probability': 3})
    print(summary_df)
    summary_df.to_csv('summary_results.csv', index=False)

    # Save data
    pokemon_stats.to_csv('capture_results.csv', index=False)
