import { render, screen } from '@testing-library/react';
import App from './App';

test('renders AI Safety Chat header', () => {
  render(<App />);
  const headerElement = screen.getByText(/AI Safety Chat/i);
  expect(headerElement).toBeInTheDocument();
});
