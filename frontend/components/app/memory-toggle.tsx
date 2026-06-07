'use client';

import { useCallback, useEffect, useState } from 'react';
import { Button } from '@/components/ui/button';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';

/** localStorage key: persists whether the agent may read/write the user's Moss profile. */
export const MEMORY_ENABLED_KEY = 'ansio_memory_enabled';
/** localStorage key: one-shot flag asking the backend to rebuild the profile on next connect. */
export const REFRESH_PROFILE_KEY = 'ansio_refresh_profile';

/** Read the memory-enabled preference (default: off). SSR-safe and never throws. */
export function readMemoryEnabled(): boolean {
  if (typeof window === 'undefined') return false;
  try {
    return window.localStorage.getItem(MEMORY_ENABLED_KEY) === '1';
  } catch {
    return false;
  }
}

/**
 * Read the one-shot "refresh profile" flag and clear it immediately (read-and-clear semantics).
 * Returns true at most once per scheduled reset. SSR-safe and never throws.
 */
export function consumeRefreshProfile(): boolean {
  if (typeof window === 'undefined') return false;
  try {
    const scheduled = window.localStorage.getItem(REFRESH_PROFILE_KEY) === '1';
    if (scheduled) {
      window.localStorage.removeItem(REFRESH_PROFILE_KEY);
    }
    return scheduled;
  } catch {
    return false;
  }
}

/**
 * Small standalone control shown near the connect button.
 * - Memory toggle: when on, the agent remembers your profile in Moss; when off it neither reads
 *   nor writes any profile.
 * - Reset profile: schedules a one-time profile rebuild that takes effect on the next connection.
 *
 * State lives entirely in localStorage so it survives reloads and is read at connect time.
 */
export function MemoryToggle({ className }: { className?: string }) {
  const [memoryEnabled, setMemoryEnabled] = useState(false);
  const [resetScheduled, setResetScheduled] = useState(false);

  // Hydrate from localStorage after mount to stay SSR-safe.
  useEffect(() => {
    setMemoryEnabled(readMemoryEnabled());
    try {
      setResetScheduled(window.localStorage.getItem(REFRESH_PROFILE_KEY) === '1');
    } catch {
      setResetScheduled(false);
    }
  }, []);

  const toggleMemory = useCallback(() => {
    setMemoryEnabled((prev) => {
      const next = !prev;
      try {
        window.localStorage.setItem(MEMORY_ENABLED_KEY, next ? '1' : '0');
      } catch {
        // Best-effort: if storage is unavailable, the UI still reflects intent for this session.
      }
      return next;
    });
  }, []);

  const scheduleReset = useCallback(() => {
    try {
      window.localStorage.setItem(REFRESH_PROFILE_KEY, '1');
    } catch {
      // Best-effort only.
    }
    setResetScheduled(true);
  }, []);

  return (
    <TooltipProvider>
      <div className={className ?? 'flex items-center justify-center gap-2'}>
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              type="button"
              size="sm"
              variant={memoryEnabled ? 'default' : 'outline'}
              aria-pressed={memoryEnabled}
              onClick={toggleMemory}
              className="rounded-full font-mono text-xs tracking-wider uppercase"
            >
              Memory: {memoryEnabled ? 'On' : 'Off'}
            </Button>
          </TooltipTrigger>
          <TooltipContent>
            On: the agent remembers your profile (stored in Moss). Off: it neither reads nor writes
            any profile.
          </TooltipContent>
        </Tooltip>

        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              type="button"
              size="sm"
              variant="ghost"
              onClick={scheduleReset}
              className="rounded-full font-mono text-xs tracking-wider uppercase"
            >
              {resetScheduled ? 'Reset scheduled' : 'Reset profile'}
            </Button>
          </TooltipTrigger>
          <TooltipContent>
            Schedules a one-time profile rebuild that takes effect on your next connection.
          </TooltipContent>
        </Tooltip>
      </div>
    </TooltipProvider>
  );
}
