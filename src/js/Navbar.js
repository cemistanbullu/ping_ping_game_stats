import { Link } from "react-router-dom";

const Navbar = () => {
  return (
    <nav className="navbar">
      <h1>Ping Pong Game Stats</h1>
      <div className="links">
        <Link to="/">Home</Link>
        <Link to="/statisticlist" style={{ 
          color: 'white', 
          backgroundColor: '#f1356d',
          borderRadius: '8px' 
        }}>Statistic</Link>
        <Link to="/otherdetail" style={{ 
          color: 'white', 
          backgroundColor: 'black',
          borderRadius: '8px' 
        }}>Other</Link>
      </div>
    </nav>
  );
}
 
export default Navbar;