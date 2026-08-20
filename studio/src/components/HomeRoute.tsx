import { useHomeSync } from './App';
import IntentScreen from './IntentScreen';

export default function HomeRoute() {
  useHomeSync();
  return <IntentScreen />;
}
