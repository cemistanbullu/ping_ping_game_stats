import Navbar from './Navbar';
import Home from './Home';
import { BrowserRouter as Router, Route, Switch } from 'react-router-dom';
import StatisticList from './StatisticList';
import StatisticDetail from './StatisticDetail';
import OtherDetail from './OtherDetail';

function App() {
  return (
    <Router>
      <div className="App">
        <Navbar />
        <div className="content">
          <Switch>
            <Route exact path="/">
              <Home />
            </Route>
            <Route path="/statisticlist">
              <StatisticList />
            </Route>
            <Route path="/statisticdetail/:id">
              <StatisticDetail />
            </Route>
            <Route path="/otherdetail">
              <OtherDetail />
            </Route>
          </Switch>
        </div>
      </div>
    </Router>
  );
}

export default App
