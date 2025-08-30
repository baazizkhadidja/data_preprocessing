


class Car:
    car_cr = 0
    pneus = 4
    def __init__(self, marque):
        Car.car_cr += 1 
        self.marque = marque
 #   def sonner(self, vitesse):
  #      self.vitesse = vitesse
   #     print(f"Pip PIP PIP je suis une {self.marque} et je roule {self.vitesse} km")
    
    



car1 = Car("Lamborghini")
car1.marque

car2 = Car("Porche")
car2.marque

Car.car_cr

#car1.sonner()
#car2.sonner()
#Car.sonner(50)

class Calculatrice:
    def __init__(self, A, B):
        self.A = A
        self.B = B
        self.r = 0
        
    def afficher(self):
        print("le resultat est " ,self.r)
        
    def addition(self):
        self.r = self.A + self.B
        self.afficher()
    
    def soustraction(self):
        self.r = self.A - self.B
        self.afficher()
    
    def multiplication(self):
        self.r = self.A * self.B
        self.afficher()
    
cal = Calculatrice(5, 6)
cal.addition()
cal.multiplication()
cal.soustraction()