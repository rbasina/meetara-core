/**
 * me²TARA Iconic Logo Component
 * 
 * Design Philosophy (Based on Trinity Architecture):
 * - Arc Reactor Core: Glowing cyan energy rings (minimal resources, maximum power)
 * - Energy Sphere: Pure energy fusion at center (unified system)
 * - Neural Network: Purple intelligence pathways (context awareness, smart routing)
 * - Geometric Markers: Trinity principles (Arc Reactor + Perplexity + Einstein)
 * 
 * Visual Identity:
 * - Professional and scientific (no romantic symbols)
 * - Clean, abstract, geometric
 * - Perplexity + Claude inspired minimalism
 * - Represents exponential intelligence amplification
 */

'use client';

import React from 'react';
import { Activity } from 'lucide-react';

// Keyframe animation for ripple effect - expand from center to edge
const shrinkExpandStyle = `
  @keyframes pulse-glow {
    0% {
      transform: translate(-50%, -50%) scale(0.2);
      opacity: 1;
    }
    100% {
      transform: translate(-50%, -50%) scale(1);
      opacity: 0.3;
    }
  }
`;

interface AnimatedLogoProps {
  size?: 'xs' | 'sm' | 'md' | 'lg' | 'xl' | '2xl';
  showText?: boolean;
  showTagline?: boolean;
  className?: string;
  variant?: 'default' | 'compact' | 'minimal';
  glowIntensity?: 'low' | 'medium' | 'high';
}

const sizeMap = {
  xs: { container: 'w-8 h-8', text: 'text-sm', icon: 'w-3 h-3' },
  sm: { container: 'w-12 h-12', text: 'text-base', icon: 'w-4 h-4' },
  md: { container: 'w-16 h-16', text: 'text-xl', icon: 'w-6 h-6' },
  lg: { container: 'w-24 h-24', text: 'text-2xl', icon: 'w-8 h-8' },
  xl: { container: 'w-32 h-32', text: 'text-4xl', icon: 'w-12 h-12' },
  '2xl': { container: 'w-48 h-48', text: 'text-5xl', icon: 'w-16 h-16' }
};

export default function AnimatedLogo({ 
  size = 'md', 
  showText = true, 
  showTagline = false,
  className = '',
  variant = 'default',
  glowIntensity = 'high' // Default to high for brightness
}: AnimatedLogoProps) {
  
  const sizes = sizeMap[size];
  
  const glowStyles = {
    low: 'shadow-lg shadow-cyan-500/30',
    medium: 'shadow-xl shadow-cyan-500/60',
    high: 'shadow-2xl shadow-cyan-500/90' // Brighter shadow
  };

  if (variant === 'minimal') {
    return <MinimalLogo size={size} showText={showText} className={className} />;
  }

  if (variant === 'compact') {
    return <CompactLogo size={size} showText={showText} className={className} />;
  }

  return (
    <>
      <style>{shrinkExpandStyle}</style>
      <div className={`flex items-center gap-4 ${className}`}>
      {/* Main Logo - Authentic me²TARA Atom Design */}
      <div className={`${sizes.container} relative`}>
        {/* Atom-like Structure - Central Glowing Sphere */}
        <div className="absolute inset-0 flex items-center justify-center">
            
            {/* Central Sphere - Smaller solid circle with pulse glow INSIDE */}
            <div className="relative w-5 h-5 flex items-center justify-center">
              {/* Solid circle container with overflow hidden to keep glow inside */}
              <div className="relative w-5 h-5 rounded-full overflow-hidden z-10"
                   style={{ 
                     background: 'radial-gradient(circle, rgba(255, 255, 255, 1) 0%, rgba(254, 243, 199, 1) 40%, rgba(253, 186, 116, 1) 100%)',
                     boxShadow: '0 0 20px rgba(255, 255, 255, 1), 0 0 40px rgba(251, 146, 60, 0.7)'
                   }}>
                {/* Pulsing glow INSIDE the circle */}
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 rounded-full"
                     style={{ 
                       background: 'radial-gradient(circle, rgba(255, 255, 255, 1) 0%, rgba(254, 243, 199, 0.9) 20%, rgba(253, 224, 171, 0.7) 40%, rgba(251, 191, 136, 0.5) 60%, rgba(251, 146, 60, 0.3) 80%, rgba(249, 115, 22, 0.1) 100%)',
                       animation: 'pulse-glow 1.5s ease-out infinite',
                       width: '100%',
                       height: '100%'
                     }}>
                </div>
              </div>
            </div>
            
            {/* Orbital Rings - Brighter rings */}
            {/* Ring 1 - Horizontal ellipse (brighter cyan) */}
            <div className="absolute inset-0 rounded-full opacity-70"
                 style={{ 
                   width: '100%',
                   height: '100%',
                   transform: 'scaleY(0.3)',
                   border: '2px solid rgba(34, 211, 238, 1)',
                   boxShadow: '0 0 8px rgba(34, 211, 238, 0.6)'
                 }} />
            
            {/* Ring 2 - Diagonal ellipse (brighter purple) - rotated 45° */}
            <div className="absolute inset-0 rounded-full opacity-65"
                 style={{ 
                   width: '100%',
                   height: '100%',
                   transform: 'rotate(45deg) scaleY(0.3)',
                   border: '2px solid rgba(168, 85, 247, 1)',
                   boxShadow: '0 0 8px rgba(168, 85, 247, 0.6)'
                 }} />
            
            {/* Ring 3 - Diagonal ellipse (brighter purple) - rotated -45° */}
            <div className="absolute inset-0 rounded-full opacity-65"
                 style={{ 
                   width: '100%',
                   height: '100%',
                   transform: 'rotate(-45deg) scaleY(0.3)',
                   border: '2px solid rgba(168, 85, 247, 1)',
                   boxShadow: '0 0 8px rgba(168, 85, 247, 0.6)'
                 }} />
            
            {/* Ring 4 - Vertical ellipse (brighter cyan) */}
            <div className="absolute inset-0 rounded-full opacity-70"
                 style={{ 
                   width: '100%',
                   height: '100%',
                   transform: 'rotate(90deg) scaleY(0.3)',
                   border: '2px solid rgba(34, 211, 238, 1)',
                   boxShadow: '0 0 8px rgba(34, 211, 238, 0.6)'
                 }} />
        </div>
      </div>
      
      {/* Brand Text - BRIGHTER gradients */}
      {showText && (
        <div className="flex flex-col gap-0">
          <h1 className={`${sizes.text} font-bold leading-none tracking-tight`}>
            <span className="bg-gradient-to-r from-cyan-400 via-blue-500 to-purple-500 bg-clip-text text-transparent drop-shadow-sm">
              me
            </span>
            <span className="bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 bg-clip-text text-transparent drop-shadow-sm">
              ²
            </span>
            <span className="bg-gradient-to-r from-purple-500 via-pink-500 to-purple-500 bg-clip-text text-transparent drop-shadow-sm">
              TARA
            </span>
          </h1>
          
          {showTagline && (
            <p className="text-xs text-gray-500 flex items-center gap-1.5 mt-0.5">
              <Activity className="w-3 h-3 text-cyan-400 animate-pulse" />
              <span>Human-AI Intelligence Fusion</span>
            </p>
          )}
        </div>
      )}
    </div>
    </>
  );
}

// Minimal Variant - Icon Only, Maximum Impact
function MinimalLogo({ size = 'md', showText, className = '' }: Omit<AnimatedLogoProps, 'showTagline' | 'variant' | 'glowIntensity'>) {
  const sizes = sizeMap[size];
  
  return (
    <div className={`flex items-center gap-3 ${className}`}>
      <div className={`${sizes.container} relative`}>
        {/* Red Square Border - Authentic me²TARA Design */}
        <div className="absolute inset-0 border-2 border-red-500 rounded-sm bg-white">
          {/* Central Glowing Sphere - BRIGHTER */}
          <div className="absolute inset-1 flex items-center justify-center">
            <div className="w-4 h-4 rounded-full bg-gradient-to-br from-cyan-200 to-cyan-400 animate-pulse shadow-lg"
                 style={{ 
                   boxShadow: '0 0 20px rgba(6, 182, 212, 1), 0 0 40px rgba(6, 182, 212, 0.8)'
                 }}>
              <div className="absolute inset-1/4 w-1 h-1 bg-white rounded-full animate-pulse"
                   style={{ boxShadow: '0 0 8px #fff' }} />
            </div>
          </div>
          
          {/* Single Orbital Ring - BRIGHTER Elliptical (Static) */}
          <div className="absolute inset-0 rounded-full border-2 border-cyan-300 opacity-100"
               style={{ 
                 transform: 'scaleY(0.6)',
                 transformOrigin: 'center'
               }} />
        </div>
      </div>
      
      {showText && (
        <span className={`${sizes.text} font-bold bg-gradient-to-r from-cyan-400 via-blue-500 to-purple-500 bg-clip-text text-transparent drop-shadow-sm`}>
          me²TARA
        </span>
      )}
    </div>
  );
}

// Compact Variant - Simplified for Small Spaces
function CompactLogo({ size = 'md', showText, className = '' }: Omit<AnimatedLogoProps, 'showTagline' | 'variant' | 'glowIntensity'>) {
  const sizes = sizeMap[size];
  
  return (
    <div className={`flex items-center gap-2 ${className}`}>
      <div className={`${sizes.container} relative`}>
        {/* Red Square Border - Authentic me²TARA Design */}
        <div className="absolute inset-0 border-2 border-red-500 rounded-sm bg-white">
          {/* Central Glowing Sphere - BRIGHTER */}
          <div className="absolute inset-1 flex items-center justify-center">
            <div className="w-3 h-3 rounded-full bg-gradient-to-br from-cyan-200 to-cyan-400 animate-pulse shadow-lg"
                 style={{ 
                   boxShadow: '0 0 15px rgba(6, 182, 212, 1), 0 0 30px rgba(6, 182, 212, 0.8)'
                 }}>
              <div className="absolute inset-1/3 w-0.5 h-0.5 bg-white rounded-full animate-pulse"
                   style={{ boxShadow: '0 0 6px #fff' }} />
            </div>
          </div>
          
          {/* Dual Orbital Rings - BRIGHTER Elliptical (Static) */}
          <div className="absolute inset-0 rounded-full border-2 border-cyan-300 opacity-100"
               style={{ 
                 transform: 'scaleY(0.6)',
                 transformOrigin: 'center'
               }} />
          <div className="absolute inset-1 rounded-full border-2 border-purple-300 opacity-90"
               style={{ 
                 transform: 'scaleY(0.7)',
                 transformOrigin: 'center'
               }} />
        </div>
      </div>
      
      {showText && (
        <div className="flex flex-col">
          <span className={`${sizes.text} font-bold leading-none bg-gradient-to-r from-cyan-400 to-purple-500 bg-clip-text text-transparent drop-shadow-sm`}>
            me²TARA
          </span>
        </div>
      )}
    </div>
  );
}