






def testi(*tupli):
    for i in tupli:
        print(i**2)
        


testi()

def testi2(**kwargs):
    for v in kwargs.values():
        print(v)


class Voiture:
    def __init__(self, marque, couleur):
        self.marque = marque 
        self.couleur = couleur
        
    def vitesse(self, vitesse):
        print(f"ma vitesse est {vitesse}")

        
v1= Voiture("Renault", "blanche")        

v1.vitesse(50)
