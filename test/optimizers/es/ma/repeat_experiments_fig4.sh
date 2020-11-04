nohup python repeat_experiments_fig4.py -bf=sphere -ndp=3 -tf=1e-14 -mg=140 >sphere_3.out 2>&1 &
nohup python repeat_experiments_fig4.py -bf=sphere -ndp=30 -tf=1e-14 -mg=600 >sphere_30.out 2>&1 &
nohup python repeat_experiments_fig4.py -bf=cigar -ndp=3 -tf=1e-15 -mg=250 >cigar_3.out 2>&1 &
nohup python repeat_experiments_fig4.py -bf=cigar -ndp=30 -tf=1e-15 -mg=1400 >cigar_30.out 2>&1 &
nohup python repeat_experiments_fig4.py -bf=discus -ndp=3 -tf=1e-15 -mg=250 >discus_3.out 2>&1 &
nohup python repeat_experiments_fig4.py -bf=discus -ndp=30 -tf=1e-15 -mg=2500 >discus_30.out 2>&1 &
nohup python repeat_experiments_fig4.py -bf=ellipsoid -ndp=3 -tf=1e-15 -mg=250 >ellipsoid_3.out 2>&1 &
nohup python repeat_experiments_fig4.py -bf=ellipsoid -ndp=30 -tf=1e-15 -mg=3000 >ellipsoid_30.out 2>&1 &
nohup python repeat_experiments_fig4.py -bf=parabolic_ridge -ndp=3 -tf=1e-1 -mg=100 >parabolic_ridge_3.out 2>&1 &
nohup python repeat_experiments_fig4.py -bf=parabolic_ridge -ndp=30 -tf=1e-2 -mg=600 >parabolic_ridge_30.out 2>&1 &
nohup python repeat_experiments_fig4.py -bf=sharp_ridge -ndp=3 -tf=1e-8 -mg=600 -nt=200 >sharp_ridge_3.out 2>&1 &
nohup python repeat_experiments_fig4.py -bf=sharp_ridge -ndp=30 -tf=1e-10 -mg=800 >sharp_ridge_30.out 2>&1 &
nohup python repeat_experiments_fig4.py -bf=rosenbrock -ndp=3 -tf=1e-10 -mg=200 -nt=100 >rosenbrock_3.out 2>&1 &
nohup python repeat_experiments_fig4.py -bf=rosenbrock -ndp=30 -tf=1e-12 -mg=3500 >rosenbrock_30.out 2>&1 &
nohup python repeat_experiments_fig4.py -bf=different_powers_beyer -ndp=3 -tf=1e-14 -mg=180 >different_powers_beyer_3.out 2>&1 &
nohup python repeat_experiments_fig4.py -bf=different_powers_beyer -ndp=30 -tf=1e-14 -mg=3000 >different_powers_beyer_30.out 2>&1 &



python repeat_experiments_fig4.py -bf=sphere -ndp=3 -mg=140 -ip=True -xn=8 -yl=-14 -yu=2 -yn=9
python repeat_experiments_fig4.py -bf=sphere -ndp=30 -mg=600 -ip=True -xn=7 -yl=-14 -yu=2 -yn=9
python repeat_experiments_fig4.py -bf=cigar -ndp=3 -mg=250 -ip=True -xn=6 -yl=-15 -yu=5 -yn=5
python repeat_experiments_fig4.py -bf=cigar -ndp=30 -mg=1400 -ip=True -xn=8 -yl=-15 -yu=5 -yn=5
python repeat_experiments_fig4.py -bf=discus -ndp=3 -mg=250 -ip=True -xn=6 -yl=-15 -yu=5 -yn=5
python repeat_experiments_fig4.py -bf=discus -ndp=30 -mg=2500 -ip=True -xn=6 -yl=-15 -yu=5 -yn=5
python repeat_experiments_fig4.py -bf=ellipsoid -ndp=3 -mg=250 -ip=True -xn=6 -yl=-15 -yu=5 -yn=5
python repeat_experiments_fig4.py -bf=ellipsoid -ndp=30 -mg=3000 -ip=True -xn=7 -yl=-15 -yu=5 -yn=5
python repeat_experiments_fig4.py -bf=parabolic_ridge -ndp=3 -mg=100 -ip=True -xn=6 -yl=-1 -yu=8 -yn=10
python repeat_experiments_fig4.py -bf=parabolic_ridge -ndp=30 -mg=600  -ip=True -xn=7 -yl=-2 -yu=10 -yn=7
python repeat_experiments_fig4.py -bf=sharp_ridge -ndp=3 -mg=600 -nt=200 -ip=True -xn=7 -yl=-8 -yu=10 -yn=10
python repeat_experiments_fig4.py -bf=sharp_ridge -ndp=30 -mg=800 -ip=True -xn=9 -yl=-10 -yu=2 -yn=7
python repeat_experiments_fig4.py -bf=rosenbrock -ndp=3 -mg=200 -nt=100 -ip=True -xn=5 -yl=-10 -yu=2 -yn=7
python repeat_experiments_fig4.py -bf=rosenbrock -ndp=30 -mg=3500 -ip=True -xn=8 -yl=-12 -yu=4 -yn=9
python repeat_experiments_fig4.py -bf=different_powers_beyer -ndp=3 -mg=180 -ip=True -xn=10 -yl=-14 -yu=2 -yn=9
python repeat_experiments_fig4.py -bf=different_powers_beyer -ndp=30 -mg=3000 -ip=True -xn=7 -yl=-14 -yu=2 -yn=9
