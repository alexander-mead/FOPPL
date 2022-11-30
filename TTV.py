# Standard imports
import torch as tc

# Third-party imports
import ttvfast

# Constants
Sun_mass = 1.9884e30    # Mass of the Sun [kg]
Earth_mass = 5.9722e24  # Mass of the Earth [kg]

# Parameters
Mstar = 1.2      # Mass of the central star [Msun]
start_time = 0.  # Simulation start time [days]
dt = 1.          # Simulation time step [days]
end_time = 1500. # Simulation end time [days]

# Fix the expected number of transits
number_of_transits = 7

#def oneplanet(mass, period, eccentricity, argument, theta, inclination, longnode):
def oneplanet(parameters):

    mass = parameters[0]
    period = parameters[1]
    eccentricity = parameters[2]
    argument = parameters[3]
    theta = parameters[4]
    inclination = parameters[5]
    longnode = parameters[6]

    #print(mass, period, eccentricity, argument, theta, inclination, longnode)

    planets = []
    planet = ttvfast.models.Planet(
        mass=mass*Earth_mass/Sun_mass,
        period=period,
        eccentricity=eccentricity,
        inclination=inclination,
        longnode=longnode,
        argument=argument,
        mean_anomaly=theta-argument,
    )
    planets.append(planet)

    # Run the TTVFast n-body simulation to get transit times
    ids, _, times, _, _ = ttvfast.ttvfast(planets, Mstar, start_time, dt, end_time)['positions']
    times = tc.tensor(times[:number_of_transits])
    return times