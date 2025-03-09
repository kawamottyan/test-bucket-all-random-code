'use client';

import { useEffect } from 'react';

interface NetworkInformation {
    effectiveType?: string;
    downlink?: number;
    rtt?: number;
    saveData?: boolean;
}

export default function SessionInitializer() {
    useEffect(() => {
        const initSession = async () => {
            try {
                const connection = (navigator as Navigator & { connection?: NetworkInformation }).connection;

                const clientInfo = {
                    timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
                    language: navigator.language,
                    screenWidth: window.screen.width,
                    screenHeight: window.screen.height,
                    colorDepth: window.screen.colorDepth,
                    windowWidth: window.innerWidth,
                    windowHeight: window.innerHeight,
                    devicePixelRatio: window.devicePixelRatio,
                    cookiesEnabled: navigator.cookieEnabled,
                    doNotTrack: navigator.doNotTrack,
                    platform: navigator.platform,
                    vendor: navigator.vendor,
                    referrer: document.referrer,
                    connectionType: connection?.effectiveType || null,
                };

                await fetch('/api/session', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(clientInfo),
                });
            } catch (error) {
                console.error('Error initializing session:', error);
            }
        };

        initSession();
    }, []);

    return null;
}