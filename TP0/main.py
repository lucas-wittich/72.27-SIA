import json
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.catching import attempt_catch
from src.pokemon import PokemonFactory, StatusEffect

# Initialize DataFrame to store results
pokemon_stats = pd.DataFrame(columns=['Pokemon', 'Pokeball', 'HP%', 'Status', 'Capture_Success', 'Capture_Probability', 'Noise', 'Catch_Rate', 'Weight', 'Type'])
noise_levels = [0.0, 0.05, 0.1, 0.15]
pokemon_configs = ['caterpie', 'snorlax']

def simulate_attempts(pokemon, pokeball, attempts=100, noise=0, catch_rate=0, weight=0, pokemon_type=''):
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
            'Weight': weight,
            'Type': pokemon_type
        })
    return results

if __name__ == '__main__':
    factory = PokemonFactory('pokemon.json')

    # Load all Pokémon data
    with open('pokemon.json', 'r') as f:
        pokemon_data = json.load(f)

    # Iterate over Pokémon configurations
    for pokemon_config in pokemon_configs:
        with open(f'configs/{pokemon_config}.json', 'r') as f:
            config = json.load(f)
            ball = config['pokeball']
            pokemon_name = config['pokemon']

            # Extract Pokémon attributes from pokemon.json
            catch_rate = pokemon_data[pokemon_name]['catch_rate']
            weight = pokemon_data[pokemon_name]['weight']
            pokemon_type = ', '.join(pokemon_data[pokemon_name]['type'])

            for level in [50, 100, 150]:
                for status in [StatusEffect.NONE, StatusEffect.SLEEP, StatusEffect.PARALYSIS]:
                    for hp in range(100, 0, -10):  # HP: 100%, 90%, ..., 10%
                        for noise in noise_levels: 
                            pokemon = factory.create(pokemon_name, level, status, hp / 100)
                            results = simulate_attempts(pokemon, ball, 10, noise, catch_rate, weight, pokemon_type)
                            df = pd.DataFrame(results)
                            pokemon_stats = pd.concat([pokemon_stats, df], ignore_index=True)

    # Compute capture success rates
    success_rates = pokemon_stats.groupby('Pokeball')['Capture_Success'].mean()

    # Save data
    pokemon_stats.to_csv('capture_results.csv', index=False)

    # Graph 1: Capture Success by Pokéball & Noise
    plt.figure(figsize=(10,6))
    sns.barplot(x='Pokeball', y='Capture_Success', hue='Noise', data=pokemon_stats)
    plt.xlabel('Pokéball Type')
    plt.ylabel('Capture Success Rate')
    plt.title('Capture Success Rate by Pokéball Type & Noise Level')
    plt.legend(title='Noise Level')
    plt.show()

    # Graph 2: Capture Probability vs. HP%
    plt.figure(figsize=(10,6))
    sns.lineplot(x='HP%', y='Capture_Probability', hue='Noise', data=pokemon_stats)
    plt.xlabel('HP%')
    plt.ylabel('Capture Probability')
    plt.title('Capture Probability vs. HP% at Different Noise Levels')
    plt.legend(title='Noise Level')
    plt.show()

    # Graph 3: Capture Probability vs. Catch Rate
    plt.figure(figsize=(10,6))
    sns.scatterplot(x='Catch_Rate', y='Capture_Probability', hue='Pokemon', data=pokemon_stats)
    plt.xlabel('Catch Rate')
    plt.ylabel('Capture Probability')
    plt.title('Capture Probability vs. Catch Rate for Different Pokémon')
    plt.legend(title='Pokemon')
    plt.show()

    # Graph 4: Capture Probability vs. Pokémon Weight
    plt.figure(figsize=(10,6))
    sns.scatterplot(x='Weight', y='Capture_Probability', hue='Pokemon', data=pokemon_stats)
    plt.xlabel('Weight')
    plt.ylabel('Capture Probability')
    plt.title('Capture Probability vs. Pokémon Weight')
    plt.legend(title='Pokemon')
    plt.show()
