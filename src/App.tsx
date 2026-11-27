import { useState } from 'react';
import {
  Car,
  Eye,
  Hand,
  Menu,
  Mic,
  PawPrint,
  UserSearch,
  X,
} from 'lucide-react';
import { CarDetectionTask } from './components/tasks/CarDetectionTask';
import { DrowsinessTask } from './components/tasks/DrowsinessTask';
import { PersonAnalysisTask } from './components/tasks/PersonAnalysisTask';
import { SignLanguageTask } from './components/tasks/SignLanguageTask';
import { VoiceEmotionTask } from './components/tasks/VoiceEmotionTask';
import { WildlifeTask } from './components/tasks/WildlifeTask';

type PageId =
  | 'car'
  | 'wildlife'
  | 'voice'
  | 'person'
  | 'sign'
  | 'drowsiness';

const NAV_ITEMS: Array<{
  id: PageId;
  label: string;
  subtitle: string;
  icon: typeof Car;
}> = [
  {
    id: 'car',
    label: 'Car Colour Detection',
    subtitle: 'Task 1 · Cars & People',
    icon: Car,
  },
  {
    id: 'wildlife',
    label: 'Wildlife Risk Intelligence',
    subtitle: 'Task 2 · Carnivore Detection',
    icon: PawPrint,
  },
  {
    id: 'voice',
    label: 'Voice Emotion Detection',
    subtitle: 'Task 3 · Speech Emotion',
    icon: Mic,
  },
  {
    id: 'person',
    label: 'Person Analysis',
    subtitle: 'Task 4 · Age & Expression',
    icon: UserSearch,
  },
  {
    id: 'sign',
    label: 'Sign Language Detection',
    subtitle: 'Task 5 · Static ASL',
    icon: Hand,
  },
  {
    id: 'drowsiness',
    label: 'Drowsiness Detection',
    subtitle: 'Task 6 · Driver Safety',
    icon: Eye,
  },
];

export default function App() {
  const [activePage, setActivePage] = useState<PageId>('car');
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const renderPage = () => {
    switch (activePage) {
      case 'car':
        return <CarDetectionTask />;
      case 'wildlife':
        return <WildlifeTask />;
      case 'voice':
        return <VoiceEmotionTask />;
      case 'person':
        return <PersonAnalysisTask />;
      case 'sign':
        return <SignLanguageTask />;
      case 'drowsiness':
        return <DrowsinessTask />;
    }
  };

  return (
    <div className="min-h-screen bg-[#07121a] text-[#edf6ff] lg:flex">
      {/* Mobile Header */}
      <header className="flex items-center justify-between border-b border-slate-800 bg-[#091620] px-4 py-3 lg:hidden">
        <div className="font-bold tracking-tight text-white">RoadSight Intelligence</div>
        <button
          type="button"
          onClick={() => setMobileMenuOpen((v) => !v)}
          className="rounded-lg border border-slate-700 p-2 text-slate-200"
          aria-label="Toggle navigation menu"
        >
          {mobileMenuOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
        </button>
      </header>

      {/* Sidebar */}
      <aside
        className={`${
          mobileMenuOpen ? 'block' : 'hidden'
        } w-full shrink-0 border-b border-slate-800 bg-[#091520] p-4 lg:block lg:min-h-screen lg:w-72 lg:border-r lg:border-b-0`}
      >
        <div className="mb-6 hidden px-2 lg:block">
          <div className="text-[11px] font-bold uppercase tracking-[0.16em] text-cyan-300">
            AI/ML Internship Suite
          </div>
          <div className="mt-1 text-xl font-bold tracking-tight text-white">
            RoadSight Intelligence
          </div>
        </div>

        <nav className="space-y-1.5">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const active = activePage === item.id;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => {
                  setActivePage(item.id);
                  setMobileMenuOpen(false);
                }}
                className={`flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left transition ${
                  active
                    ? 'border border-cyan-400/40 bg-cyan-500/15 text-white'
                    : 'border border-transparent text-slate-300 hover:bg-slate-800/60 hover:text-white'
                }`}
              >
                <Icon className={`h-4 w-4 shrink-0 ${active ? 'text-cyan-300' : 'text-slate-400'}`} />
                <div className="min-w-0">
                  <div className="truncate text-sm font-medium">{item.label}</div>
                  <div className="truncate text-[11px] text-slate-400">{item.subtitle}</div>
                </div>
              </button>
            );
          })}
        </nav>
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-x-hidden p-4 sm:p-6 lg:p-8">
        <div className="mx-auto max-w-6xl">{renderPage()}</div>
      </main>
    </div>
  );
}
