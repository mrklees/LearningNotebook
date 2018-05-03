package example

// Defining a object (instance of a class) which extends the 
// trait Greeting.  
object Hello extends Greeting with App {
  def main() {
    println(greeting)
  } 
  main()
}

trait Greeting {
  lazy val greeting: String = "Hello World!"
}